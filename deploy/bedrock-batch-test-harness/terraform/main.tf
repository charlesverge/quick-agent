data "aws_caller_identity" "current" {}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

locals {
  s3_bucket_name = "${var.bucket_name_prefix}-${data.aws_caller_identity.current.account_id}-${var.aws_region}-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket" "bedrock_batch" {
  bucket        = local.s3_bucket_name
  force_destroy = var.force_destroy_bucket
}

resource "aws_s3_bucket_public_access_block" "bedrock_batch" {
  bucket = aws_s3_bucket.bedrock_batch.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "bedrock_batch" {
  bucket = aws_s3_bucket.bedrock_batch.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_iam_role" "bedrock_batch_role" {
  name = var.bedrock_batch_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_batch_role_policy" {
  name = "${var.bedrock_batch_role_name}-policy"
  role = aws_iam_role.bedrock_batch_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3InputOutput"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.bedrock_batch.arn,
          "${aws_s3_bucket.bedrock_batch.arn}/${var.input_prefix}/*",
          "${aws_s3_bucket.bedrock_batch.arn}/${var.output_prefix}/*"
        ]
      },
      {
        Sid    = "AllowInvokeModelBatch"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.model_id}",
          "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
          "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:application-inference-profile/*"
        ]
      }
    ]
  })
}
