variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "system_profile" {
  type     = string
  default  = null
  nullable = true
}

variable "deployer_user_name" {
  type    = string
  default = "quick-agent-bedrock-deployer"
}

variable "deployer_profile_name" {
  type    = string
  default = "quick-agent-bedrock-deployer"
}

variable "bucket_name_prefix" {
  type    = string
  default = "jobs-agent-bedrock-batch"
}

variable "bedrock_batch_role_name" {
  type    = string
  default = "quick-agent-bedrock-batch-role"
}

variable "model_id" {
  type    = string
  default = "qwen.qwen3-coder-30b-a3b-v1:0"
}
