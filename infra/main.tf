provider "google" {
  project     = "graphic-armor-339522"
  region      = "asia-northeast1-b"
}

resource "google_storage_bucket" "checkpoints" {
  name          = "florians_results"
  location      = "asia-northeast1"
  force_destroy = true  # Forces bucket deletion even if it contains objects
}

resource "google_compute_instance" "florians-deeplearning" {
  boot_disk {
    auto_delete = true
    device_name = "florians-deeplearning"

    initialize_params {
      image = "projects/ml-images/global/images/c2-deeplearning-pytorch-2-2-cu121-v20240514-debian-11-py310"
      size  = 50
      type  = "pd-ssd"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/graphic-armor-339522/zones/asia-east1-c/acceleratorTypes/nvidia-tesla-v100"
  }

  labels = {
    goog-ec-src = "vm_add-tf"
  }

  machine_type = "n1-highcpu-8"
  name         = "florians-deeplearning"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/graphic-armor-339522/regions/asia-east1/subnetworks/default"
  }

  scheduling {
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
  }

  service_account {
    email  = "619899876982-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = "asia-east1-c"
}
