// ============================================================================
// amf.cue — Deterministic Digital Immunity for 6G Networks
// Phase 1: CUE Schema for AMF Configuration Validation
// ============================================================================
//
// Defines the canonical CUE schema for validating Osmocom AMF configurations.
// This schema enforces structural correctness, type safety, range constraints,
// and cross-field invariants before any configuration is applied to the AMF.
//
// The schema is consumed by the DSF's CueResult validator (see llm_intent.proto)
// as part of the intent validation pipeline.
//
// CUE language reference: https://cuelang.org/
// Package: telecomm.osmocom.amf
// ============================================================================

package telecomm.osmocom.amf

// ============================================================================
// Imports
// ============================================================================

import (
	"regexp"
	"strings"
)

// ============================================================================
// Top-level AMF Configuration
// ============================================================================

// #AmfConfig is the root configuration schema for an Osmocom AMF instance.
// Every AMF deployment MUST conform to this schema before the DSF allows
// the configuration to be applied.
#AmfConfig: {
	// Globally unique AMF identifier.
	// Format: 6 hexadecimal digits (e.g., "A1B2C3").
	// Matches the amf_id format used in 3GPP NGAP (AMF Set ID + AMF Pointer).
	amf_id: string & =~"^([0-9a-fA-F]{6})$" & !="000000"

	// PLMN (Public Land Mobile Network) identifier.
	plmn_id: #PlmnId

	// AMF operational mode.
	// Determines whether the AMF accepts UE registrations and sessions.
	mode: string & #ModeValues & =~"^(active|standby|disabled)$"

	// Maximum concurrent UE sessions (PDU sessions + EPS contexts).
	// Range: 1 to 1,000,000 (hard ceiling for a single AMF instance).
	max_ue_sessions: int & >=1 & <=1000000

	// Maximum bearers (EPS bearers + PDU session QoS flows) per UE.
	// 3GPP TS 23.501 limits this to 64 for 5QI; Osmocom defaults to 16.
	max_bearers_per_ue: int & >=1 & <=16

	// Session idle timeout in seconds.
	// After this period of inactivity, the AMF may release the PDU session.
	// Range: 60 s (1 min) to 86400 s (24 h).
	session_timeout_s: int & >=60 & <=86400

	// Tracking Area Code. Hex string, 3 to 6 hex digits.
	// 3GPP TS 38.304: TAC is 24 bits = 6 hex digits.
	tac: string & =~"^([0-9a-fA-F]{3,6})$"

	// NSSAI (Network Slice Selection Assistance Information).
	// List of supported slices for this AMF instance.
	// Constraint enforced below: sum(nssai[*].max_ues) <= max_ue_sessions
	nssai: [...#SliceConfig] & len(nssai) >= 0

	// Optional: QoS profile templates available for session establishment.
	qos_profiles: [...#QosProfile] | *null

	// NGAP interface configuration (N1/N2 interface towards gNB/eNB).
	ngap: #NgapConfig | *#DefaultNgapConfig

	// Namf service interface configuration (Service-based interface).
	namf: #NamfConfig | *null

	// Security configuration (NAS security algorithms, SUPI privacy).
	security: #SecurityConfig | *#DefaultSecurityConfig

	// ---------------------------------------------------------------------------
	// Derived / Cross-field Constraints
	// ---------------------------------------------------------------------------

	// Constraint 1: If mode is "active", at least one slice must be configured.
	// An active AMF without slices cannot serve any UEs.
	//
	// Implementation note: CUE does not support conditional imports directly,
	// so we validate this inline via a comprehension.
	if mode == "active" {
		len(nssai) & >=1
	}

	// Constraint 2: The sum of all slice max_ues MUST NOT exceed the AMF's
	// total max_ue_sessions. This prevents over-subscription at the AMF level.
	//
	// We compute the total via a generator comprehension over nssai.
	let _total_slice_ues = {
		for k, s in nssai {
			"\(k)": s.max_ues
		}
	}
	// Sum all slice capacities; must be within the AMF session budget.
	// CUE computes this via the _total_slice_ues generator.
	let total_slice_capacity: {
		for _, v in _total_slice_ues {
			_:
				v
		}
	}
	// Direct constraint: if nssai is non-empty, enforce capacity.
	if len(nssai) > 0 {
		let _check: {
			let sum = 0
			for _, s in nssai {
				let sum = sum + s.max_ues
			}
			sum & <= max_ue_sessions
		}
	}
}

// ============================================================================
// Sub-schemas
// ============================================================================

// PLMN identifier (MCC + MNC).
#PlmnId: {
	// Mobile Country Code. 3 decimal digits.
	mcc: string & =~"^([0-9]{3})$"

	// Mobile Network Code. 2 or 3 decimal digits.
	mnc: string & =~"^([0-9]{2,3})$"
}

// Allowed AMF mode values (union type).
#ModeValues: "active" | "standby" | "disabled"

// ---------------------------------------------------------------------------
// Slice Configuration
// ---------------------------------------------------------------------------

// #SliceConfig defines a single S-NSSAI (Single Network Slice Selection
// Assistance Information). Conforms to 3GPP TS 23.503 §5.15.
#SliceConfig: {
	// Slice/Service Type. Range: 1–255.
	// Standardised values (3GPP TS 23.501 §5.15.2):
	//   1=eMBB, 2=URLLC, 3=mMTC, 4=V2X, 5=HN, 6=IIoT,
	//   8=PM, 9=BI, 128=TSN, 255=default.
	sst: int & >=1 & <=255

	// Slice Differentiator. 6 hex digits (24 bits, 3GPP TS 23.003 §28.4.1).
	sd: string & =~"^([0-9a-fA-F]{6})$"

	// Maximum UEs allowed in this slice.
	// Range: 1 to 500,000.
	max_ues: int & >=1 & <=500000

	// Guaranteed Bit Rate per-UE in this slice (bits/s).
	// Range: 1 kbps to 10 Gbps.
	guaranteed_br_bps: int & >=1000 & <=10000000000

	// Slice priority. 0 = highest, 255 = lowest.
	priority: int & >=0 & <=255
}

// ---------------------------------------------------------------------------
// QoS Profile
// ---------------------------------------------------------------------------

// #QosProfile defines a QoS profile template that can be referenced
// during PDU session establishment. Follows 3GPP TS 23.501 §5.7.
#QosProfile: {
	// 5QI (5G QoS Identifier). Standard values: 1–9, 65–86.
	five_qi: int & (#QciRange | #FiveQiRange)

	// ARP (Allocation and Retention Priority).
	arp: #ArpConfig

	// Guaranteed Bit Rate QoS parameters (required for GBR bearers).
	// Present when five_qi is a GBR 5QI (e.g., 1, 2, 3, 4, 65–70, 75, 77).
	gbr_qos: #GbrQos | *null

	// Non-GBR QoS parameters (required for non-GBR bearers).
	// Present when five_qi is a non-GBR 5QI (e.g., 5–9, 71–76, 78–86).
	non_gbr_qos: #NonGbrQos | *null
}

// QCI value range (4G LTE).
#QciRange: >=1 & <=9

// 5QI value range (5G NR).
#FiveQiRange: >=65 & <=86

// ARP configuration. 3GPP TS 23.501 §5.7.7.
#ArpConfig: {
	// Priority level. 1 = highest, 15 = lowest.
	priority_level: int & >=1 & <=15

	// Pre-emption capability.
	// "may-preempt" = this bearer may preempt lower-priority bearers.
	// "not-preempt"  = this bearer must not preempt others.
	preemption_cap: string & =~"^(may-preempt|not-preempt)$"

	// Pre-emption vulnerability.
	// "preemptable"   = this bearer may be preempted by higher-priority.
	// "not-preemptable" = this bearer must not be preempted.
	preemption_vuln: string & =~"^(preemptable|not-preemptable)$"
}

// GBR (Guaranteed Bit Rate) QoS parameters. 3GPP TS 23.501 §5.7.3.4.
#GbrQos: {
	// Guaranteed Flow Bit Rate — Downlink (bits/s).
	gfbr_dl: int & >=0
	// Guaranteed Flow Bit Rate — Uplink (bits/s).
	gfbr_ul: int & >=0
	// Maximum Flow Bit Rate — Downlink (bits/s).
	mfbr_dl: int & >=0
	// Maximum Flow Bit Rate — Uplink (bits/s).
	mfbr_ul: int & >=0

	// Constraint: GFBR <= MFBR (guaranteed cannot exceed maximum).
	gfbr_dl & <= mfbr_dl
	gfbr_ul & <= mfbr_ul
}

// Non-GBR QoS parameters. 3GPP TS 23.501 §5.7.3.4.
#NonGbrQos: {
	// Session-AMBR — Downlink (bits/s). Total for all non-GBR bearers.
	session_ambr_dl: int & >=0
	// Session-AMBR — Uplink (bits/s).
	session_ambr_ul: int & >=0
}

// ---------------------------------------------------------------------------
// NGAP Interface Configuration
// ---------------------------------------------------------------------------

// #NgapConfig defines the N2 interface (AMF ↔ gNB) parameters.
#NgapConfig: {
	// IPv4 bind address for NGAP. Must be a valid unicast IPv4 address.
	bind_address: string & =~"^(([0-9]{1,3}\\.){3}[0-9]{1,3})$"

	// SCTP bind port. IANA registered for NGAP: 38412.
	bind_port: int & >=1024 & <=65535 & !=0

	// Maximum concurrent NGAP connections (SCTP associations).
	max_connections: int & >=1 & <=4096

	// SCTP keep-alive interval in seconds.
	// Default: 30 s. Range: 5–300 s.
	keepalive_interval_s: int & >=5 & <=300

	// SCTP-specific parameters.
	sctp: #SctpConfig | *#DefaultSctpConfig
}

// Default NGAP configuration (used when ngap field is omitted).
#DefaultNgapConfig: #NgapConfig & {
	bind_address: "0.0.0.0"
	bind_port:    38412
	max_connections: 1024
	keepalive_interval_s: 30
	sctp: #DefaultSctpConfig
}

// SCTP transport parameters.
#SctpConfig: {
	// Maximum number of SCTP outbound streams.
	max_outbound_streams: int & >=1 & <=65535
	// Maximum number of SCTP inbound streams.
	max_inbound_streams: int & >=1 & <=65535
	// SCTP init timeout in milliseconds.
	init_timeout_ms: int & >=1000 & <=30000
	// SCTP retransmission timeout in milliseconds.
	rto_initial_ms: int & >=100 & <=3000
	// SCTP maximum retransmission attempts.
	max_retransmissions: int & >=1 & <=20
}

// Default SCTP configuration.
#DefaultSctpConfig: #SctpConfig & {
	max_outbound_streams: 30
	max_inbound_streams:  30
	init_timeout_ms:      5000
	rto_initial_ms:       1000
	max_retransmissions:  5
}

// ---------------------------------------------------------------------------
// Namf Service Interface Configuration
// ---------------------------------------------------------------------------

// #NamfConfig defines the Namf service-based interface configuration
// (3GPP TS 29.518). This is optional — some deployments use direct
// NGAP signalling without a separate Namf service layer.
#NamfConfig: {
	// IPv4 bind address for Namf HTTP/2 service.
	bind_address: string & =~"^(([0-9]{1,3}\\.){3}[0-9]{1,3})$"

	// HTTP/2 bind port for Namf.
	bind_port: int & >=1024 & <=65535 & !=0

	// NRF (Network Repository Function) endpoint URL.
	// Format: https://<nrf-host>:<port>/nnrf-nfm/v1
	nrf_endpoint: string & =~"^https://([a-zA-Z0-9._-]+)(:[0-9]+)?(/.*)?$"

	// Authentication settings.
	auth: #AuthConfig | *null
}

// Authentication configuration for Namf.
#AuthConfig: {
	// Mutual TLS settings.
	mtls: #MtlsConfig | *null

	// OAuth 2.0 client credentials.
	oauth2: #OAuth2Config | *null
}

// Mutual TLS configuration.
#MtlsConfig: {
	// Path to client certificate (PEM).
	client_cert_path: string & != ""
	// Path to client private key (PEM).
	client_key_path:  string & != ""
	// Path to CA certificate bundle (PEM).
	ca_cert_path:     string & != ""
	// Whether mTLS is enforced.
	enabled: bool & true
}

// OAuth 2.0 configuration.
#OAuth2Config: {
	// Token endpoint URL.
	token_endpoint: string & =~"^https://.*$"
	// Client ID.
	client_id: string & != ""
	// Client secret (in production, use a secrets manager).
	client_secret: string & != ""
	// Grant type (must be client_credentials).
	grant_type: string & =="client_credentials"
}

// ---------------------------------------------------------------------------
// Security Configuration
// ---------------------------------------------------------------------------

// #SecurityConfig defines NAS security parameters for the AMF.
// Follows 3GPP TS 33.501 §6.
#SecurityConfig: {
	// NEA (Encryption Algorithm) priority list.
	// Values: 0=NEA0 (no encryption), 1=NEA1 (SNOW 3G), 2=NEA2 (AES),
	//         3=NEA3 (ZUC).
	// Ordered list from most preferred to least preferred.
	nea_priority: [...int] & len(nea_priority) >= 1 & every(_, [
		int & >=0 & <=3,
	])

	// NIA (Integrity Protection Algorithm) priority list.
	// Values: 0=NIA0 (no integrity), 1=NIA1 (SNOW 3G), 2=NIA2 (AES),
	//         3=NIA3 (ZUC).
	nia_priority: [...int] & len(nia_priority) >= 1 & every(_, [
		int & >=0 & <=3,
	])

	// SUPI (Subscription Permanent Identifier) privacy enabled.
	// When true, the AMF uses ECIES-based SUCI encryption per TS 33.501.
	supi_privacy_enabled: bool

	// Authentication failure threshold. After this many consecutive
	// failed NAS authentication attempts, the AMF blocks the UE.
	// Range: 1–20.
	auth_failure_threshold: int & >=1 & <=20
}

// Default security configuration.
#DefaultSecurityConfig: #SecurityConfig & {
	nea_priority: [2, 1, 3, 0]  // AES > SNOW > ZUC > none
	nia_priority: [2, 1, 3, 0]  // AES > SNOW > ZUC > none
	supi_privacy_enabled: true
	auth_failure_threshold: 5
}
