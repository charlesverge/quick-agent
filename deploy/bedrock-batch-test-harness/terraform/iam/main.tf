data "aws_caller_identity" "current" {}

resource "aws_iam_user" "deployer" {
  name = var.deployer_user_name
}

resource "aws_iam_access_key" "deployer" {
  user = aws_iam_user.deployer.name
}

resource "aws_iam_user_policy" "deployer" {
  name = "${var.deployer_user_name}-bedrock-batch-deploy"
  user = aws_iam_user.deployer.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketLifecycle"
        Effect = "Allow"
        Action = [
          "s3:CreateBucket",
          "s3:DeleteBucket",
          "s3:GetBucketAcl",
          "s3:GetBucketCORS",
          "s3:GetBucketLocation",
          "s3:GetBucketWebsite",
          "s3:GetBucketVersioning",
          "s3:ListBucket",
          "s3:GetAccelerateConfiguration",
          "s3:PutBucketVersioning",
          "s3:Get*",
          "s3:PutBucketPublicAccessBlock",
          "s3:GetBucketPublicAccessBlock",
          "s3:DeleteBucketPolicy",
          "s3:PutBucketPolicy",
          "iam:ListInstanceProfilesForRole"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name_prefix}-*"
        ]
      },
      {
        Sid    = "S3ObjectReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name_prefix}-*/*"
        ]
      },
      {
        Sid    = "S3BucketPolicyRead"
        Effect = "Allow"
        Action = [
          "s3:GetBucketPolicy"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name_prefix}-*"
        ]
      },
      {
        Sid    = "IamRoleManagementForHarness"
        Effect = "Allow"
        Action = [
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:GetRole",
          "iam:UpdateAssumeRolePolicy",
          "iam:PutRolePolicy",
          "iam:DeleteRolePolicy",
          "iam:GetRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:ListInstanceProfilesForRole",
          "iam:TagRole",
          "iam:UntagRole",
          "iam:PassRole"
        ]
        Resource = [
          "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.bedrock_batch_role_name}"
        ]
      },
      {
        Sid    = "BedrockBatchJobExecution"
        Effect = "Allow"
        Action = [
          "bedrock:CreateModelInvocationJob",
          "bedrock:GetModelInvocationJob",
          "bedrock:ListModelInvocationJobs",
          "bedrock:StopModelInvocationJob"
        ]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-invocation-job/*",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.model_id}",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/*",
          "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
          "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:application-inference-profile/*"
        ]
      }
    ]
  })
}
