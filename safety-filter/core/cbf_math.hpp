// ============================================================================
// cbf_math.hpp — Deterministic Digital Immunity for 6G Networks
// Phase 2: DSF Core — CBF/Lyapunov Mathematical Engine (Header)
// ============================================================================
//
// Defines the core C++ types and engine for evaluating Control Barrier
// Functions (CBF), Lyapunov stability conditions, and computing Lie
// derivatives for network state / control input pairs.
//
// Mathematical background:
//   A Control Barrier Function h(x) enforces forward invariance of the
//   safe set C = {x : h(x) >= 0} under control input u:
//
//       sup_u [ L_f*h(x) + L_g*h(x)*u ] >= -γ(h(x))
//
//   where L_f*h is the Lie derivative of h along the drift f,
//         L_g*h is the Lie derivative of h along the control g,
//         γ is an extended class-K function.
//
//   A Lyapunov function V(x) certifies asymptotic stability of the
//   reference x_ref:
//
//       V̇(x, u) = dV/dt <= -α·V(x)
//
//   where α > 0 is the decay rate.
//
// Thread safety: This engine is NOT thread-safe by default. External
// synchronisation is required for concurrent evaluation. However,
// evaluate_all_barriers and evaluate_lyapunov are independent reads
// on internal state and safe to call concurrently if no barriers are
// being added simultaneously.
//
// ============================================================================

#pragma once

// ============================================================================
// Includes
// ============================================================================

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

// ============================================================================
// Logging
// ============================================================================

// Simple timestamp-based logger using std::cerr. In production, this can
// be replaced with spdlog or any other logging framework. The macro
// interface ensures zero overhead when logging is disabled.

#ifndef CBF_LOG_LEVEL
#define CBF_LOG_LEVEL 2  // 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE
#endif

namespace cbf::core::log {

/// Returns a timestamp string for log lines.
inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

}  // namespace cbf::core::log

#define CBF_LOG(level, msg)                                                    \
    do {                                                                       \
        if (CBF_LOG_LEVEL >= level) {                                          \
            std::cerr << "[" << cbf::core::log::timestamp() << "] "            \
                      << "[" << level << "] " << __FILE__ << ":" << __LINE__   \
                      << " — " << msg << std::endl;                            \
        }                                                                      \
    } while (0)

#define CBF_LOG_ERROR(msg) CBF_LOG(1, "ERROR: " << msg)
#define CBF_LOG_WARN(msg)  CBF_LOG(2, "WARN:  " << msg)
#define CBF_LOG_INFO(msg)  CBF_LOG(3, "INFO:  " << msg)
#define CBF_LOG_DEBUG(msg) CBF_LOG(4, "DEBUG: " << msg)
#define CBF_LOG_TRACE(msg) CBF_LOG(5, "TRACE: " << msg)

// ============================================================================
// Namespace
// ============================================================================

namespace cbf::core {

// ============================================================================
// Result Types
// ============================================================================

/// Result of evaluating a single Control Barrier Function constraint.
/// Encapsulates the barrier value, class-K function value, and the
/// computed safety margin for the given state-control pair.
struct CbfResult {
    /// True if the CBF constraint is satisfied:
    ///   L_f*h(x) + L_g*h(x)*u >= -γ(h(x))
    bool passed = false;

    /// Current barrier function value h(x).
    /// Positive means the state is within the safe set.
    double cbf_value = 0.0;

    /// Class-K function value γ(h(x)).
    /// For extended class-K: γ(h) = k·h (when k=1, γ(h) = h).
    double gamma_value = 0.0;

    /// Safety margin:
    ///   margin = [L_f*h + L_g*h*u] - [-γ(h)]
    ///           = L_f*h + L_g*h*u + γ(h)
    /// Positive margin indicates the barrier is satisfied.
    double margin = 0.0;

    /// Barrier identifier (e.g., "tx_power_ceiling").
    std::string barrier_id;
};

/// Result of evaluating the Lyapunov stability condition.
/// Checks whether the system is converging to the reference state.
struct LyapunovResult {
    /// True if V̇(x, u) <= -α·V(x) (stable).
    bool passed = false;

    /// Current Lyapunov function value V(x, x_ref).
    double V = 0.0;

    /// Computed time derivative V̇(x, u).
    double dV = 0.0;

    /// Decay rate constant α.
    double alpha = 0.0;

    /// Stability threshold: -α·V(x).
    /// V̇ must be <= this value for stability.
    double threshold = 0.0;
};

// ============================================================================
// Core Data Structures
// ============================================================================

/// Network state representation. The state vector x contains the
/// dynamically relevant quantities (e.g., [tx_power, cpu_util, temp, ...]).
/// The params map carries static metadata (e.g., element_id, timestamps)
/// that does not participate in CBF evaluation but is useful for logging.
struct NetworkState {
    /// State vector (n-dimensional). Must match the state_dim of CbfEngine.
    Eigen::VectorXd x;

    /// Wall-clock timestamp of this state measurement (epoch seconds).
    double timestamp = 0.0;

    /// Auxiliary parameters (metadata, not used in CBF computation).
    /// Keys: parameter names (e.g., "element_id", "access_mode").
    std::map<std::string, double> params;
};

/// A single Control Barrier Function definition.
///
/// The barrier function h(x) defines the safe set C = {x : h(x) >= 0}.
/// The gradient ∇h(x) is used for Lie derivative computation.
/// The class-K function γ(h) defines the required descent rate.
struct BarrierFunction {
    /// Unique barrier identifier (e.g., "tx_power_ceiling").
    std::string id;

    /// Barrier function h : R^n → R.
    /// Returns h(x). The safe set is {x : h(x) >= 0}.
    std::function<double(const Eigen::VectorXd&)> h;

    /// Gradient of the barrier function ∇h : R^n → R^n (as row vector).
    /// Returns ∂h/∂x evaluated at x.
    std::function<Eigen::RowVectorXd(const Eigen::VectorXd&)> grad_h;

    /// Extended class-K function γ : R → R.
    /// γ(h) >= 0 for h >= 0, γ(0) = 0, strictly increasing.
    /// Common choice: γ(h) = k·h with k > 0.
    std::function<double(double)> gamma;

    /// Human-readable barrier expression (for logging / audit).
    /// Example: "h(x) = 43 - tx_power >= 0".
    std::string expr;
};

/// Lyapunov stability configuration.
///
/// Defines the Lyapunov function V(x, x_ref) and its time derivative
/// V̇(x, x_ref, u) for stability certification.
struct LyapunovConfig {
    /// Decay rate constant α > 0.
    /// The system satisfies V̇ <= -α·V for exponential convergence.
    double alpha = 0.1;

    /// Reference state x_ref. The Lyapunov function measures deviation
    /// from this equilibrium (or desired operating point).
    Eigen::VectorXd x_ref;

    /// Lyapunov function V : R^n × R^n → R_{>=0}.
    /// Takes (current_state, reference_state) and returns V >= 0.
    /// V(x, x_ref) = 0 iff x = x_ref.
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> V;

    /// Time derivative of the Lyapunov function V̇ : R^n × R^n × R^m → R.
    /// Takes (current_state, reference_state, control_input) and returns dV/dt.
    /// Computed via Lie derivative or numerical finite differences.
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&,
                         const Eigen::VectorXd&)> dVdt;

    /// Whether this Lyapunov config has been fully initialised.
    bool is_configured() const {
        return V && dVdt && x_ref.size() > 0 && alpha > 0.0;
    }
};

// ============================================================================
// CBF Engine
// ============================================================================

/// Core CBF/Lyapunov evaluation engine.
///
/// This engine maintains a set of barrier functions and an optional
/// Lyapunov configuration. Given a NetworkState and a control input u,
/// it evaluates all safety constraints and returns the results.
///
/// Usage pattern:
///   CbfEngine engine(state_dim);
///   engine.add_barrier({...});
///   engine.set_lyapunov({...});
///   auto result = engine.evaluate(state, u, current_time);
///
class CbfEngine {
public:
    /// Construct a CBF engine for a fixed state dimension.
    /// @param state_dim Dimension of the state vector x (must be > 0).
    /// @throws std::invalid_argument if state_dim <= 0.
    explicit CbfEngine(int state_dim);

    /// Destructor.
    ~CbfEngine() = default;

    // Non-copyable, non-movable (barriers hold std::function references).
    CbfEngine(const CbfEngine&) = delete;
    CbfEngine& operator=(const CbfEngine&) = delete;
    CbfEngine(CbfEngine&&) = delete;
    CbfEngine& operator=(CbfEngine&&) = delete;

    /// Register a barrier function with the engine.
    /// @param barrier The BarrierFunction to add.
    /// @throws std::invalid_argument if barrier.h or barrier.grad_h is null.
    void add_barrier(BarrierFunction barrier);

    /// Configure the Lyapunov stability monitor.
    /// @param config The Lyapunov configuration.
    /// @throws std::invalid_argument if config is not fully initialised.
    void set_lyapunov(LyapunovConfig config);

    /// Evaluate ALL barrier functions and return results for each.
    /// @param state Current network state.
    /// @param u Control input vector.
    /// @return Vector of CbfResult, one per registered barrier.
    /// @throws std::invalid_argument if dimensions mismatch.
    std::vector<CbfResult> evaluate_all_barriers(
        const NetworkState& state,
        const Eigen::VectorXd& u);

    /// Evaluate the Lyapunov stability condition.
    /// @param state Current network state.
    /// @param u Control input vector.
    /// @return LyapunovResult with stability check outcome.
    /// @throws std::invalid_argument if Lyapunov is not configured.
    LyapunovResult evaluate_lyapunov(
        const NetworkState& state,
        const Eigen::VectorXd& u);

    /// Combined evaluation: runs CBF barriers first, then Lyapunov.
    /// Returns the FIRST violation found (CBF has priority over Lyapunov
    /// because barrier violations are safety-critical, while Lyapunov
    /// violations indicate sub-optimal convergence).
    ///
    /// @param state Current network state.
    /// @param u Control input vector.
    /// @param current_time Wall-clock time in seconds (for logging).
    /// @return std::variant<CbfResult, LyapunovResult> — the first violation,
    ///         or std::monostate if everything passes (not used here —
    ///         caller should check the full results separately).
    std::variant<CbfResult, LyapunovResult> evaluate(
        const NetworkState& state,
        const Eigen::VectorXd& u,
        double current_time);

    /// Get the number of registered barriers.
    /// @return Number of barrier functions.
    [[nodiscard]] size_t barrier_count() const noexcept { return barriers_.size(); }

    /// Check if Lyapunov is configured.
    /// @return True if a Lyapunov configuration is active.
    [[nodiscard]] bool has_lyapunov() const noexcept { return lyap_config_.is_configured(); }

    /// Get the state dimension.
    /// @return State vector dimension.
    [[nodiscard]] int state_dim() const noexcept { return state_dim_; }

private:
    /// Registered barrier functions.
    std::vector<BarrierFunction> barriers_;

    /// Lyapunov configuration (optional).
    LyapunovConfig lyap_config_;

    /// State vector dimension (fixed at construction).
    int state_dim_;

    /// Validate that the state vector has the correct dimension.
    /// @param x State vector to validate.
    /// @throws std::invalid_argument on dimension mismatch.
    void validate_state_dim(const Eigen::VectorXd& x) const;
};

// ============================================================================
// Lie Derivative Helpers
// ============================================================================

/// Compute the Lie derivative of a scalar function h along a vector
/// field f using the gradient: L_f*h(x) = ∇h(x) · f(x).
///
/// This function takes the gradient directly (computed analytically or
/// via finite differences) and the drift vector f.
///
/// @param grad_h Gradient of h evaluated at x (row vector).
/// @param f Drift vector f(x).
/// @return L_f*h(x) = grad_h · f (scalar).
inline double compute_lie_derivative_from_grad(
    const Eigen::RowVectorXd& grad_h,
    const Eigen::VectorXd& f) {
    return (grad_h * f).value();
}

/// Compute the Lie derivative of h along the control direction g,
/// weighted by the control input u: L_g*h(x) * u = ∇h(x) · (g(x)*u).
///
/// For affine systems: ẋ = f(x) + g(x)*u, the control Lie derivative
/// is L_g*h(x)·u = ∇h(x) · [G(x) · u] where G(x) is the control matrix.
///
/// @param grad_h Gradient of h evaluated at x (row vector).
/// @param g Control effectiveness matrix g(x) (n × m).
/// @param u Control input vector (m × 1).
/// @return L_g*h(x) * u (scalar).
inline double compute_control_lie_derivative(
    const Eigen::RowVectorXd& grad_h,
    const Eigen::MatrixXd& g,
    const Eigen::VectorXd& u) {
    return (grad_h * g * u).value();
}

/// Compute the gradient of a scalar function using central finite differences.
/// Useful when analytical gradients are not available.
///
/// ∂h/∂x_i ≈ [h(x + ε·e_i) - h(x - ε·e_i)] / (2·ε)
///
/// where e_i is the i-th standard basis vector and ε is the step size.
///
/// @param h Scalar function h : R^n → R.
/// @param x Evaluation point.
/// @param epsilon Finite difference step size (default: 1e-7).
/// @return Gradient ∇h(x) as a row vector.
inline Eigen::RowVectorXd compute_gradient_central_difference(
    const std::function<double(const Eigen::VectorXd&)>& h,
    const Eigen::VectorXd& x,
    double epsilon = 1e-7) {
    const int n = static_cast<int>(x.size());
    Eigen::RowVectorXd grad(n);

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus(i) += epsilon;
        x_minus(i) -= epsilon;

        // Clamp to avoid numerical issues at boundaries.
        x_plus(i) = std::max(x_plus(i), -1e15);
        x_minus(i) = std::min(x_minus(i), 1e15);

        const double h_plus = h(x_plus);
        const double h_minus = h(x_minus);

        // Check for NaN/Inf in the finite difference.
        if (!std::isfinite(h_plus) || !std::isfinite(h_minus)) {
            // Fallback to forward difference.
            Eigen::VectorXd x_fwd = x;
            x_fwd(i) += epsilon;
            const double h_fwd = h(x_fwd);
            grad(i) = (h_fwd - h(x)) / epsilon;
            CBF_LOG_WARN("Central difference NaN/Inf at index " << i
                         << ", falling back to forward difference");
        } else {
            grad(i) = (h_plus - h_minus) / (2.0 * epsilon);
        }
    }

    return grad;
}

/// Compute the gradient of a scalar function using forward finite
/// differences (lower accuracy but avoids calling h twice per dimension).
///
/// @param h Scalar function h : R^n → R.
/// @param x Evaluation point.
/// @param epsilon Finite difference step size (default: 1e-7).
/// @return Gradient ∇h(x) as a row vector.
inline Eigen::RowVectorXd compute_gradient_forward_difference(
    const std::function<double(const Eigen::VectorXd&)>& h,
    const Eigen::VectorXd& x,
    double epsilon = 1e-7) {
    const int n = static_cast<int>(x.size());
    Eigen::RowVectorXd grad(n);
    const double h_x = h(x);

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x_plus = x;
        x_plus(i) += epsilon;
        grad(i) = (h(x_plus) - h_x) / epsilon;
    }

    return grad;
}

// ============================================================================
// Utility: Vector to string (for logging)
// ============================================================================

/// Convert an Eigen vector to a human-readable string.
/// @param v The vector to stringify.
/// @param precision Number of decimal places (default: 4).
/// @return String representation.
inline std::string eigen_to_string(const Eigen::VectorXd& v, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << "[";
    for (int i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << v(i);
    }
    oss << "]";
    return oss.str();
}

}  // namespace cbf::core
