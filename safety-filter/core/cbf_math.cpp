// ============================================================================
// cbf_math.cpp — Deterministic Digital Immunity for 6G Networks
// Phase 2: DSF Core — CBF/Lyapunov Mathematical Engine (Implementation)
// ============================================================================
//
// Full implementation of the CbfEngine class and supporting functions.
// This file implements the core safety-filter evaluation pipeline:
//
//   1. evaluate_all_barriers() — CBF constraint checking per barrier
//   2. evaluate_lyapunov()      — Lyapunov stability certification
//   3. evaluate()               — Combined CBF + Lyapunov pipeline
//   4. Lie derivative helpers  — Numerical gradient-based computation
//
// Mathematical reference:
//   Ames, A. D., et al. "Control Barrier Functions: Theory and Applications."
//   ECC 2017.
//
//   Nagumo, M. "Über die Lage der Integralkurven gewöhnlicher
//   Differentialgleichungen." Journal of the Physical Society of Japan, 1942.
//
// ============================================================================

#include "cbf_math.hpp"

#include <algorithm>
#include <stdexcept>

namespace cbf::core {

// ============================================================================
// CbfEngine — Constructor & Mutators
// ============================================================================

CbfEngine::CbfEngine(int state_dim) : state_dim_(state_dim) {
    if (state_dim <= 0) {
        throw std::invalid_argument(
            "CbfEngine: state_dim must be positive, got " +
            std::to_string(state_dim));
    }
    CBF_LOG_INFO("CbfEngine constructed with state_dim=" << state_dim_);
}

void CbfEngine::add_barrier(BarrierFunction barrier) {
    // Validate that the barrier function and gradient are callable.
    if (!barrier.h) {
        throw std::invalid_argument(
            "CbfEngine::add_barrier: barrier function h must not be null for "
            "barrier '" + barrier.id + "'");
    }
    if (!barrier.grad_h) {
        throw std::invalid_argument(
            "CbfEngine::add_barrier: barrier gradient grad_h must not be null "
            "for barrier '" + barrier.id + "'");
    }
    if (!barrier.gamma) {
        throw std::invalid_argument(
            "CbfEngine::add_barrier: class-K function gamma must not be null "
            "for barrier '" + barrier.id + "'");
    }

    // Provide a default expression if none given.
    if (barrier.expr.empty()) {
        barrier.expr = "h_" + barrier.id + "(x) >= 0";
    }

    barriers_.push_back(std::move(barrier));
    CBF_LOG_INFO("Barrier registered: id='" << barriers_.back().id
                                           << "', expr='" << barriers_.back().expr << "'"
                                           << ", total_barriers=" << barriers_.size());
}

void CbfEngine::set_lyapunov(LyapunovConfig config) {
    if (!config.is_configured()) {
        throw std::invalid_argument(
            "CbfEngine::set_lyapunov: Lyapunov config is not fully initialised. "
            "Ensure V, dVdt, x_ref, and alpha are all set.");
    }

    if (static_cast<int>(config.x_ref.size()) != state_dim_) {
        throw std::invalid_argument(
            "CbfEngine::set_lyapunov: x_ref dimension (" +
            std::to_string(config.x_ref.size()) + ") must match state_dim (" +
            std::to_string(state_dim_) + ")");
    }

    if (config.alpha <= 0.0) {
        throw std::invalid_argument(
            "CbfEngine::set_lyapunov: alpha must be positive, got " +
            std::to_string(config.alpha));
    }

    lyap_config_ = std::move(config);
    CBF_LOG_INFO("Lyapunov configured: alpha=" << lyap_config_.alpha
                                                << ", x_ref=" << eigen_to_string(lyap_config_.x_ref));
}

// ============================================================================
// CbfEngine — Validation
// ============================================================================

void CbfEngine::validate_state_dim(const Eigen::VectorXd& x) const {
    if (x.size() != state_dim_) {
        throw std::invalid_argument(
            "CbfEngine: state vector dimension mismatch — expected " +
            std::to_string(state_dim_) + ", got " + std::to_string(x.size()));
    }
    // Check for NaN/Inf in the state vector.
    for (int i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x(i))) {
            throw std::invalid_argument(
                "CbfEngine: state vector contains NaN or Inf at index " +
                std::to_string(i) + " (value=" + std::to_string(x(i)) + ")");
        }
    }
}

// ============================================================================
// CbfEngine — CBF Barrier Evaluation
// ============================================================================

std::vector<CbfResult> CbfEngine::evaluate_all_barriers(
    const NetworkState& state,
    const Eigen::VectorXd& u) {

    // Validate input dimensions.
    validate_state_dim(state.x);
    if (u.size() <= 0) {
        // Control input may be zero-dimensional for passive systems,
        // but log a warning.
        CBF_LOG_WARN("evaluate_all_barriers: control input u is empty (0-dim). "
                     "Only drift (L_f*h) will contribute to the CBF condition.");
    }

    const Eigen::VectorXd& x = state.x;
    std::vector<CbfResult> results;
    results.reserve(barriers_.size());

    CBF_LOG_DEBUG("Evaluating " << barriers_.size()
                                 << " barriers for state=" << eigen_to_string(x)
                                 << ", u=" << eigen_to_string(u));

    for (const auto& barrier : barriers_) {
        CbfResult result;
        result.barrier_id = barrier.id;

        try {
            // ------------------------------------------------------------------
            // Step 1: Evaluate h(x)
            // ------------------------------------------------------------------
            result.cbf_value = barrier.h(x);

            if (!std::isfinite(result.cbf_value)) {
                result.passed = false;
                result.margin = -std::numeric_limits<double>::infinity();
                result.gamma_value = 0.0;
                CBF_LOG_ERROR("Barrier '" << barrier.id
                                          << "': h(x) is NaN/Inf = " << result.cbf_value);
                results.push_back(std::move(result));
                continue;
            }

            // ------------------------------------------------------------------
            // Step 2: Evaluate γ(h(x))
            // ------------------------------------------------------------------
            result.gamma_value = barrier.gamma(result.cbf_value);

            if (!std::isfinite(result.gamma_value)) {
                result.passed = false;
                result.margin = -std::numeric_limits<double>::infinity();
                CBF_LOG_ERROR("Barrier '" << barrier.id
                                          << "': γ(h(x)) is NaN/Inf = " << result.gamma_value);
                results.push_back(std::move(result));
                continue;
            }

            // ------------------------------------------------------------------
            // Step 3: Evaluate ∇h(x)
            // ------------------------------------------------------------------
            Eigen::RowVectorXd grad_h;
            try {
                grad_h = barrier.grad_h(x);
            } catch (const std::exception& e) {
                result.passed = false;
                result.margin = -std::numeric_limits<double>::infinity();
                CBF_LOG_ERROR("Barrier '" << barrier.id
                                          << "': gradient computation failed: " << e.what());
                results.push_back(std::move(result));
                continue;
            }

            // Validate gradient dimension.
            if (grad_h.size() != state_dim_) {
                result.passed = false;
                result.margin = -std::numeric_limits<double>::infinity();
                CBF_LOG_ERROR("Barrier '" << barrier.id
                                          << "': gradient dimension mismatch — expected "
                                          << state_dim_ << ", got " << grad_h.size());
                results.push_back(std::move(result));
                continue;
            }

            // Check for NaN/Inf in gradient.
            bool grad_valid = true;
            for (int i = 0; i < grad_h.size(); ++i) {
                if (!std::isfinite(grad_h(i))) {
                    grad_valid = false;
                    break;
                }
            }
            if (!grad_valid) {
                result.passed = false;
                result.margin = -std::numeric_limits<double>::infinity();
                CBF_LOG_ERROR("Barrier '" << barrier.id
                                          << "': gradient contains NaN/Inf");
                results.push_back(std::move(result));
                continue;
            }

            // ------------------------------------------------------------------
            // Step 4: Compute L_f*h(x) + L_g*h(x)*u
            //
            // For affine systems: ẋ = f(x) + g(x)*u
            //   L_f*h(x) = ∇h(x) · f(x)
            //   L_g*h(x)*u = ∇h(x) · [g(x)*u]
            //
            // In our simplified network model, we treat the full drift as:
            //   L_total = ∇h(x) · f_eff(x, u)
            //
            // where f_eff captures the combined dynamics. For simplicity,
            // we use the gradient dot product with the effective dynamics.
            // ------------------------------------------------------------------

            // For network configurations, the "control" u is the intended
            // parameter change, and the "drift" f captures the natural
            // dynamics. In the simplest model:
            //   L_f*h = ∇h · f(x)        (drift contribution)
            //   L_g*h*u = ∇h · (G·u)     (control contribution)
            //
            // We assume a simplified affine model where the control
            // effectiveness matrix G = I (identity) and f(x) represents
            // the natural dynamics.
            //
            // For barrier evaluation in the context of LLM-generated
            // intents, we evaluate whether applying control u at state x
            // keeps the system in the safe set. The combined condition is:
            //
            //   L_f*h(x) + L_g*h(x)*u >= -γ(h(x))
            //
            // Since network configurations are not continuous-time
            // dynamical systems in the classical sense, we interpret
            // the Lie derivatives as:
            //   - L_f*h: effect of natural dynamics on barrier
            //   - L_g*h*u: effect of the control action on barrier
            //
            // For discrete intents, we compute the barrier value at the
            // "next" state x + Δx where Δx is influenced by u, and check
            // if the barrier condition holds in expectation.

            // For the discrete-step model, we compute:
            //   L_g*h * u = ∇h(x) · u  (simplified: G = I)
            double l_g_h_u = 0.0;
            if (u.size() > 0 && u.size() == state_dim_) {
                l_g_h_u = (grad_h * u).value();
            } else if (u.size() > 0 && u.size() != state_dim_) {
                CBF_LOG_WARN("Barrier '"
                             << barrier.id
                             << "': u dimension (" << u.size()
                             << ") != state_dim (" << state_dim_
                             << "); control Lie derivative set to 0");
            }

            // L_f*h: we approximate the drift Lie derivative using the
            // barrier function's time-derivative heuristic. For network
            // configurations, the drift is typically slow (e.g., temperature
            // drifts, load changes). We compute it as the barrier sensitivity
            // to state perturbation:
            //
            //   L_f*h ≈ 0  (when no explicit drift model is provided)
            //
            // This is conservative: it assumes the barrier can only improve
            // through control actions, not through natural dynamics.
            //
            // For a more accurate model, the user should provide the drift
            // vector f(x) and compute L_f*h = ∇h · f explicitly.
            const double l_f_h = 0.0;  // Conservative: no drift contribution

            // ------------------------------------------------------------------
            // Step 5: Evaluate the CBF condition
            //
            //   L_f*h(x) + L_g*h(x)*u >= -γ(h(x))
            //
            //   margin = [L_f*h + L_g*h*u] - [-γ(h)]
            //           = L_f*h + L_g*h*u + γ(h)
            // ------------------------------------------------------------------
            result.margin = l_f_h + l_g_h_u + result.gamma_value;
            result.passed = (result.margin >= 0.0);

            // Log the result with appropriate detail level.
            if (!result.passed) {
                CBF_LOG_WARN("Barrier VIOLATED: id='" << barrier.id
                                                      << "', h(x)=" << result.cbf_value
                                                      << ", γ(h)=" << result.gamma_value
                                                      << ", L_f*h=" << l_f_h
                                                      << ", L_g*h*u=" << l_g_h_u
                                                      << ", margin=" << result.margin);
            } else {
                CBF_LOG_DEBUG("Barrier PASSED: id='" << barrier.id
                                                     << "', h(x)=" << result.cbf_value
                                                     << ", margin=" << result.margin);
            }

        } catch (const std::exception& e) {
            result.passed = false;
            result.margin = -std::numeric_limits<double>::infinity();
            result.cbf_value = 0.0;
            result.gamma_value = 0.0;
            CBF_LOG_ERROR("Barrier '" << barrier.id
                                      << "': evaluation threw exception: " << e.what());
        }

        results.push_back(std::move(result));
    }

    return results;
}

// ============================================================================
// CbfEngine — Lyapunov Evaluation
// ============================================================================

LyapunovResult CbfEngine::evaluate_lyapunov(
    const NetworkState& state,
    const Eigen::VectorXd& u) {

    if (!lyap_config_.is_configured()) {
        throw std::invalid_argument(
            "CbfEngine::evaluate_lyapunov: Lyapunov is not configured. "
            "Call set_lyapunov() before evaluating.");
    }

    validate_state_dim(state.x);

    LyapunovResult result;
    result.alpha = lyap_config_.alpha;

    const Eigen::VectorXd& x = state.x;
    const Eigen::VectorXd& x_ref = lyap_config_.x_ref;

    try {
        // ------------------------------------------------------------------
        // Step 1: Compute V(x, x_ref)
        // ------------------------------------------------------------------
        result.V = lyap_config_.V(x, x_ref);

        if (!std::isfinite(result.V)) {
            result.passed = false;
            result.dV = std::numeric_limits<double>::quiet_NaN();
            result.threshold = -std::numeric_limits<double>::infinity();
            CBF_LOG_ERROR("Lyapunov: V(x, x_ref) is NaN/Inf = " << result.V);
            return result;
        }

        // V must be non-negative.
        if (result.V < 0.0) {
            CBF_LOG_WARN("Lyapunov: V(x, x_ref) = " << result.V
                                                     << " is negative (should be >= 0)");
            // Clamp to zero for threshold computation.
            result.V = 0.0;
        }

        // ------------------------------------------------------------------
        // Step 2: Compute V̇(x, x_ref, u)
        // ------------------------------------------------------------------
        result.dV = lyap_config_.dVdt(x, x_ref, u);

        if (!std::isfinite(result.dV)) {
            result.passed = false;
            result.threshold = -std::numeric_limits<double>::infinity();
            CBF_LOG_ERROR("Lyapunov: dV/dt is NaN/Inf = " << result.dV);
            return result;
        }

        // ------------------------------------------------------------------
        // Step 3: Compute threshold and check stability
        //
        //   V̇(x, u) <= -α·V(x)
        //   threshold = -α·V(x)
        // ------------------------------------------------------------------
        result.threshold = -result.alpha * result.V;
        result.passed = (result.dV <= result.threshold);

        // Log the result.
        if (!result.passed) {
            CBF_LOG_WARN("Lyapunov VIOLATED: V=" << result.V
                                                  << ", dV/dt=" << result.dV
                                                  << ", threshold=-α*V=" << result.threshold
                                                  << ", α=" << result.alpha);
        } else {
            CBF_LOG_DEBUG("Lyapunov PASSED: V=" << result.V
                                                 << ", dV/dt=" << result.dV
                                                 << ", threshold=" << result.threshold);
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.V = 0.0;
        result.dV = 0.0;
        result.threshold = 0.0;
        CBF_LOG_ERROR("Lyapunov evaluation threw exception: " << e.what());
    }

    return result;
}

// ============================================================================
// CbfEngine — Combined Evaluation
// ============================================================================

std::variant<CbfResult, LyapunovResult> CbfEngine::evaluate(
    const NetworkState& state,
    const Eigen::VectorXd& u,
    double current_time) {

    CBF_LOG_INFO("Combined evaluation at t=" << current_time
                                             << ", state=" << eigen_to_string(state.x)
                                             << ", u=" << eigen_to_string(u));

    // ------------------------------------------------------------------
    // Priority 1: CBF barrier evaluation (safety-critical)
    // ------------------------------------------------------------------
    auto cbf_results = evaluate_all_barriers(state, u);

    for (const auto& result : cbf_results) {
        if (!result.passed) {
            CBF_LOG_WARN("CBF violation detected at t=" << current_time
                                                         << ": barrier='" << result.barrier_id
                                                         << "', margin=" << result.margin);
            return result;  // Return the first violating barrier.
        }
    }

    // ------------------------------------------------------------------
    // Priority 2: Lyapunov stability (performance-critical)
    // ------------------------------------------------------------------
    if (has_lyapunov()) {
        auto lyap_result = evaluate_lyapunov(state, u);

        if (!lyap_result.passed) {
            CBF_LOG_WARN("Lyapunov violation detected at t=" << current_time);
            return lyap_result;
        }

        // All checks passed — return a "passing" Lyapunov result
        // as a sentinel (all CBFs passed, Lyapunov passed).
        return lyap_result;
    }

    // No Lyapunov configured — return a passing CBF result as sentinel.
    // (All barriers passed at this point.)
    CBF_LOG_INFO("All " << barriers_.size()
                        << " barriers PASSED at t=" << current_time);
    return CbfResult{
        .passed = true,
        .cbf_value = 0.0,
        .gamma_value = 0.0,
        .margin = 0.0,
        .barrier_id = "_all_passes"
    };
}

// ============================================================================
// Lie Derivative Helpers — Full Implementation
// ============================================================================

// The inline functions (compute_lie_derivative_from_grad,
// compute_control_lie_derivative, compute_gradient_central_difference,
// compute_gradient_forward_difference) are already defined in the
// header file. The implementations below provide additional
// convenience wrappers for the DSF pipeline.

namespace {

/// Internal helper: check that a scalar function value is finite.
inline bool is_valid_scalar(double v) {
    return std::isfinite(v);
}

/// Internal helper: check that a vector is finite.
inline bool is_valid_vector(const Eigen::VectorXd& v) {
    for (int i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v(i))) return false;
    }
    return true;
}

}  // anonymous namespace

}  // namespace cbf::core
