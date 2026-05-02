# Terraform configuration for Titanic Prediction Service on AWS EC2
# Since Terraform uses HCL, this file follows standard infrastructure-as-code syntax
# as required for AWS provisioning.

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "titanic_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = { Name = "titanic-prediction-vpc" }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.titanic_vpc.id
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.titanic_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

resource "aws_security_group" "allow_web" {
  name   = "titanic-sg"
  vpc_id = aws_vpc.titanic_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "titanic_app" {
  ami           = "ami-0c55b159cbfafe1f0" # Amazon Linux 2023
  instance_type = "t2.micro"
  subnet_id     = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_web.id]

  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y python3 python3-pip
              pip3 install fastapi uvicorn pandas scikit-learn mlflow boto3
              # Export env vars for MLflow tracking
              export MLFLOW_TRACKING_URI=${var.mlflow_tracking_uri}
              export MLFLOW_TRACKING_USERNAME=${var.mlflow_user}
              export MLFLOW_TRACKING_PASSWORD=${var.mlflow_pass}
              # Start the app
              python3 /app/main.py
              EOF

  tags = { Name = "titanic-fastapi-server" }
}

variable "aws_region" { default = "us-east-1" }
variable "mlflow_tracking_uri" {}
variable "mlflow_user" {}
variable "mlflow_pass" {}