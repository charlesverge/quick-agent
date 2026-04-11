variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "aws_profile" {
  type    = string
  default = "quick-agent-bedrock-deployer"
}

variable "bucket_name_prefix" {
  type    = string
  default = "quick-agent-bedrock-batch"
}

variable "model_id" {
  type    = string
  default = "qwen.qwen3-coder-30b-a3b-v1:0"
}

variable "input_prefix" {
  type    = string
  default = "quick-agent-bedrock/input"
}

variable "output_prefix" {
  type    = string
  default = "quick-agent-bedrock/output"
}

variable "force_destroy_bucket" {
  type    = bool
  default = false
}

variable "bedrock_batch_role_name" {
  type    = string
  default = "quick-agent-bedrock-batch-role"
}
