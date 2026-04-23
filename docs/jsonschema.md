# JSON Schema Notes

This project uses provider-specific structured-output behavior.

For Amazon Bedrock, `boto3` and `botocore` validate request shape against the SDK service model, but they do not provide a built-in validator for Bedrock's supported JSON Schema subset. That means the SDK can catch malformed request payloads, but Bedrock-specific schema rules such as unsupported keywords, external `$ref`, recursive references, or `additionalProperties` values other than `false` are enforced by Amazon Bedrock, not by a separate boto3 helper.

The Bedrock structured-output behavior and supported JSON Schema subset are documented in the AWS blog post [Structured outputs on Amazon Bedrock: Schema-compliant AI responses](https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/). That article explicitly lists the supported features and unsupported features used by this project when validating Bedrock schemas.

Supported JSON Schema features for Bedrock:

- Basic types: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null`
- `enum` with strings, numbers, booleans, or `null`
- `const`, `anyOf`, `allOf`
- Internal `$ref`, `$def`, and `definitions`
- String formats: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid`
- `minItems` with values `0` or `1`

Unsupported JSON Schema features for Bedrock:

- Recursive schemas
- External `$ref` references
- Numerical constraints such as `minimum`, `maximum`, and `multipleOf`
- String constraints such as `minLength` and `maxLength`
- `additionalProperties` set to anything other than `false`

For agentic workflows on Bedrock, tool definitions should use `strict: true` so tool parameters must match the input schema exactly. The same AWS article covers this strict tool-use mode.
