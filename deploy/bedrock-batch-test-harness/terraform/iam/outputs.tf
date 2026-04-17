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

output "bedrock_console_user_name" {
  value = aws_iam_user.bedrock_console.name
}

output "bedrock_console_initial_password" {
  value     = aws_iam_user_login_profile.bedrock_console.password
  sensitive = true
}

output "bedrock_console_initial_password_visible" {
  value = nonsensitive(aws_iam_user_login_profile.bedrock_console.password)
}

output "bedrock_console_sign_in_url" {
  value = "https://${data.aws_caller_identity.current.account_id}.signin.aws.amazon.com/console"
}
