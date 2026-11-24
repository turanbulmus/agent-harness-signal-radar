resource "google_service_account" "radar_runner" {
  account_id   = "${var.service_name}-runner"
  display_name = "Agent Harness Signal Radar Runtime Service Account"
  project      = var.project_id
}

locals {
  runner_project_roles = toset([
    "roles/aiplatform.user",
    "roles/cloudtrace.agent",
    "roles/dlp.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
}

resource "google_project_iam_member" "radar_runner_roles" {
  for_each = local.runner_project_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.radar_runner.email}"
}

resource "google_storage_bucket_iam_member" "artifact_bucket_admin" {
  bucket = google_storage_bucket.agent_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.radar_runner.email}"
}
