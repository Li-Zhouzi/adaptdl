REGION="us-east-1"
INSTANCE_ID="i-0d37bb8b3cf6d3813"
ACCOUNT_ID="399790253372"
IMG="399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:cifar10"
PLATFORM="linux/amd64"

aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunDocument" \
  --timeout-seconds 5400 \
  --comment "Pull $IMG using SSM executeAwsApi for ECR auth" \
  --parameters '{
    "documentParameters": "[ \
      { \
        \"action\":\"aws:executeAwsApi\", \
        \"name\":\"getToken\", \
        \"inputs\":{ \
          \"Service\":\"ecr\", \
          \"Api\":\"GetAuthorizationToken\", \
          \"RegistryIds\":[\"'"$ACCOUNT_ID"'\"] \
        } \
      }, \
      { \
        \"action\":\"aws:runShellScript\", \
        \"name\":\"pullWithCtr\", \
        \"inputs\":{ \
          \"runCommand\": [ \
            \"set -euo pipefail\", \
            \"TOKEN_B64={{ getToken.authorizationData[0].authorizationToken }}\", \
            \"PASS=$(echo \\\"$TOKEN_B64\\\" | base64 -d | cut -d: -f2-)\", \
            \"apiclient exec admin -- sheltie bash -lc 'ctr -a /run/containerd/containerd.sock -n k8s.io images pull --platform '"$PLATFORM"' '"$IMG"' --user AWS:${PASS}'\" \
          ] \
        } \
      } \
    ]"
  }' \
  --query "Command.CommandId" \
  --output text
