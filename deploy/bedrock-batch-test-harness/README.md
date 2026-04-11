# Bedrock Batch Test Harness

Production-like test harness for quick-agent Bedrock batch workflows.

## Location and structure

Harness root:

- `deploy/bedrock-batch-test-harness`

Required stage modules:

- `run.py` (primary entrypoint)
- `setup.py`
- `execution.py`
- `verify.py`
- `settings.py` (shared configuration loader)
- `fixtures/default.json` (parameterized inputs)

Supporting assets:

- `execution.py` (Bedrock job execution/import logic)
- `terraform/` (AWS setup)
- `terraform/iam/` (developer IAM credentials bootstrap)

## Prerequisites

- AWS CLI authenticated as a system-level IAM principal for IAM bootstrap
- Terraform `>= 1.5`
- Python environment with project dependencies installed
- Bedrock access to configured model (`qwen.qwen3-coder-30b-a3b-v1:0` by default)

## IAM bootstrap (system credentials)

Run this once with a system-level profile that is allowed to create IAM users and access keys.

`system_profile` means an existing AWS CLI profile name on the local machine with elevated IAM permissions (for example `default`, `admin`, or another organization admin profile). Terraform uses this profile only for the IAM bootstrap stack in `terraform/iam`.

List available local profiles:

```bash
aws configure list-profiles
```

Then use one of those profile names as `system_profile`:

```bash
terraform apply -var='system_profile=admin'
```

```bash
cd deploy/bedrock-batch-test-harness/terraform/iam
terraform init
terraform apply -var='system_profile=YOUR_SYSTEM_PROFILE'
```

This flow is designed for two different machines:

1. system admin machine: create IAM user/access key
1. developer machine: configure local AWS profile from handed-off credentials

On the system admin machine, export a handoff file after `terraform apply`:

```bash
cat > quick-agent-bedrock-deployer.credentials <<EOF
[quick-agent-bedrock-deployer]
aws_access_key_id=$(terraform output -raw deployer_access_key_id)
aws_secret_access_key=$(terraform output -raw deployer_secret_access_key)
region=$(terraform output -raw aws_region)
EOF
```

Share that file with the developer through a secure channel (for example enterprise password manager secure note, encrypted file transfer, or secrets manager). Do not commit this file to git.

On the developer machine, merge the profile into `~/.aws/credentials`:

```bash
cat quick-agent-bedrock-deployer.credentials >> ~/.aws/credentials
```

Or set values explicitly:

```bash
aws configure set aws_access_key_id "<KEY_ID>" --profile quick-agent-bedrock-deployer
aws configure set aws_secret_access_key "<SECRET_KEY>" --profile quick-agent-bedrock-deployer
aws configure set region "us-east-1" --profile quick-agent-bedrock-deployer
```

## Provision deployment resources (developer credentials)

```bash
cd deploy/bedrock-batch-test-harness/terraform
terraform init
terraform apply
```

Optional override:

```bash
terraform apply -var='aws_region=us-east-1' -var='model_id=qwen.qwen3-next-80b-a3b'
```

Terraform authentication for this stack uses the developer profile created above. The provider in `terraform/versions.tf` is configured with:

- `profile = var.aws_profile` (default: `quick-agent-bedrock-deployer`)
- `region = var.aws_region`

Override profile when needed:

```bash
terraform apply -var='aws_profile=quick-agent-bedrock-deployer'
```

## Single-command harness execution

From repo root:

```bash
python deploy/bedrock-batch-test-harness/run.py
```

This runs:

1. setup
1. execution
1. verification

Stage responsibilities:

- setup: resolve Terraform-backed runtime settings, generate input JSONL, upload input JSONL to S3
- execution: submit Bedrock batch job, wait/download results, run import-results
- verification: strict JSONL/result-count checks

Strict verification checks:

- expected files exist:
  - `deploy/bedrock-batch-test-harness/safe/bedrock/input-100.jsonl`
  - `deploy/bedrock-batch-test-harness/safe/bedrock/output-100.jsonl`
  - `deploy/bedrock-batch-test-harness/safe/bedrock/import-outcomes-100.jsonl`
- files are valid JSONL
- input/output row counts equal configured `count` (default `100`)
- import outcome row count equals output row count

## Stage-selective runs

Setup only:

```bash
python deploy/bedrock-batch-test-harness/run.py --setup
```

Execution only:

```bash
python deploy/bedrock-batch-test-harness/run.py --execute
```

Verification only:

```bash
python deploy/bedrock-batch-test-harness/run.py --verify
```

Lifecycle flags:

```bash
python deploy/bedrock-batch-test-harness/run.py --no-tear-down
python deploy/bedrock-batch-test-harness/run.py --cleanup
```

`--cleanup` removes runtime state, logs, and generated local JSONL artifacts.

## Configuration source

Harness stages read configuration from:

- `deploy/bedrock-batch-test-harness/fixtures/default.json`
- Terraform outputs resolved during setup (`runtime/runtime_settings.json`)

To change runtime behavior, update the fixture file instead of passing many command-line options.
