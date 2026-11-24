resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret" "github_pat_token" {
  secret_id = "github-pat-token"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret" "digest_webhook_url" {
  secret_id = "digest-webhook-url"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_iam_member" "gemini_api_key_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.radar_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "github_token_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.github_pat_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.radar_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "webhook_url_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.digest_webhook_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.radar_runner.email}"
}
