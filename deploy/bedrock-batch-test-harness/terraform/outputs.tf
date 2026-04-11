output "aws_region" {
  value = var.aws_region
}

output "aws_profile" {
  value = var.aws_profile
}

output "model_id" {
  value = var.model_id
}

output "bedrock_batch_role_arn" {
  value = aws_iam_role.bedrock_batch_role.arn
}

output "s3_bucket_name" {
  value = aws_s3_bucket.bedrock_batch.bucket
}

output "s3_input_uri" {
  value = "s3://${aws_s3_bucket.bedrock_batch.bucket}/${var.input_prefix}/input-100.jsonl"
}

output "s3_output_uri" {
  value = "s3://${aws_s3_bucket.bedrock_batch.bucket}/${var.output_prefix}/"
}
