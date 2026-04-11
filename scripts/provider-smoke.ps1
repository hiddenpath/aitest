param(
    [string]$BaseUrl = "http://127.0.0.1:3000",
    [string]$Prompt = "Reply with exactly: pong"
)

$ErrorActionPreference = "Stop"

function Invoke-NonStream {
    param(
        [string]$Url,
        [string]$ModelId,
        [string]$PromptText
    )

    $body = @{
        user_id = "smoke-user"
        session_id = "smoke-$($ModelId -replace '[^a-zA-Z0-9]+','-')"
        message = $PromptText
        model_id = $ModelId
    } | ConvertTo-Json -Compress

    try {
        $resp = Invoke-RestMethod -Method Post -Uri "$Url/chat" -ContentType "application/json" -Body $body -TimeoutSec 90
        return [pscustomobject]@{
            ok = [bool]$resp.ok
            content = [string]$resp.content
            completion_tokens = if ($resp.usage) { [int]$resp.usage.completion_tokens } else { -1 }
            error = ""
        }
    } catch {
        return [pscustomobject]@{
            ok = $false
            content = ""
            completion_tokens = -1
            error = $_.Exception.Message
        }
    }
}

function Invoke-Stream {
    param(
        [string]$Url,
        [string]$ModelId,
        [string]$PromptText
    )

    $tmp = Join-Path $env:TEMP ("aitest-smoke-" + [guid]::NewGuid().ToString() + ".json")
    try {
        $payload = @{
            user_id = "smoke-user"
            session_id = "smoke-$($ModelId -replace '[^a-zA-Z0-9]+','-')-stream"
            message = $PromptText
            model_id = $ModelId
        } | ConvertTo-Json -Compress
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($tmp, $payload, $utf8NoBom)

        $cmd = 'curl.exe --noproxy 127.0.0.1,localhost -sS -N -H "Content-Type: application/json" --data-binary "@' + $tmp + '" --max-time 90 "' + $Url + '/chat/stream"'
        $stream = & cmd.exe /c $cmd
        $streamText = ($stream -join "`n")
        return [pscustomobject]@{
            has_delta = [bool]($streamText -match '"type":"delta"')
            has_usage = [bool]($streamText -match '"type":"usage"')
            has_done = [bool]($streamText -match '"type":"done"')
            excerpt = if ($streamText.Length -gt 240) { $streamText.Substring(0, 240) } else { $streamText }
        }
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

$modelsResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/models" -TimeoutSec 30
$results = @()

foreach ($model in $modelsResp.models) {
    $non = Invoke-NonStream -Url $BaseUrl -ModelId $model.id -PromptText $Prompt
    $stream = Invoke-Stream -Url $BaseUrl -ModelId $model.id -PromptText $Prompt
    $results += [pscustomobject]@{
        model = $model.id
        display_name = $model.display_name
        nonstream_ok = $non.ok
        nonstream_content = $non.content
        nonstream_completion_tokens = $non.completion_tokens
        nonstream_error = $non.error
        stream_has_delta = $stream.has_delta
        stream_has_usage = $stream.has_usage
        stream_has_done = $stream.has_done
        stream_excerpt = $stream.excerpt
    }
}

$results | ConvertTo-Json -Depth 5
