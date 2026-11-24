variable "project_id" {
  description = "Google Cloud project ID hosting the Agent Harness Signal Radar deployment."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Agent Engine, and Artifact storage."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment label (e.g. dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "service_name" {
  description = "Base name for Cloud Run service, service account, and artifact resources."
  type        = string
  default     = "harness-signal-radar"
}

variable "container_image" {
  description = "Container image URI built from the root Dockerfile."
  type        = string
  default     = "us-central1-docker.pkg.dev/example-project/agents/harness-signal-radar:latest"
}

variable "reasoning_model" {
  description = "Deep reasoning Gemini model ID for planning and synthesis."
  type        = string
  default     = "gemini-2.5-pro"
}

variable "fast_model" {
  description = "Low-latency Gemini model ID for parallel extraction and self-eval."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "enable_cloud_dlp" {
  description = "Enable Google Cloud Sensitive Data Protection API scrubbing."
  type        = bool
  default     = true
}

variable "model_armor_template_id" {
  description = "Optional Model Armor template resource name for prompt inspection."
  type        = string
  default     = ""
}
