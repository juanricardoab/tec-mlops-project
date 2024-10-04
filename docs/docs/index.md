# tec-mlops-project documentation!

## Description

Project Repository Team 2 - MLOps.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://bucket_test_mlops/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://bucket_test_mlops/data/` to `data/`.


