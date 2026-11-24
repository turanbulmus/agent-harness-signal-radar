output "cloud_run_service_uri" {
  description = "Deployed Cloud Run ADK API server endpoint URI."
  value       = google_cloud_run_v2_service.radar_api.uri
}

output "runner_service_account_email" {
  description = "Least-privilege service account used by the runtime."
  value       = google_service_account.radar_runner.email
}

output "artifact_bucket_url" {
  description = "GCS staging and research artifact bucket URL."
  value       = google_storage_bucket.agent_artifacts.url
}

output "secret_manager_secret_ids" {
  description = "Secret Manager secret resource names to populate out-of-band."
  value = [
    google_secret_manager_secret.gemini_api_key.id,
    google_secret_manager_secret.github_pat_token.id,
    google_secret_manager_secret.digest_webhook_url.id,
  ]
}
