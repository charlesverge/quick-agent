output "aws_region" {
  value = var.aws_region
}

output "deployer_profile_name" {
  value = var.deployer_profile_name
}

output "deployer_user_name" {
  value = aws_iam_user.deployer.name
}

output "deployer_access_key_id" {
  value = aws_iam_access_key.deployer.id
}

output "deployer_secret_access_key" {
  value     = aws_iam_access_key.deployer.secret
  sensitive = true
}
