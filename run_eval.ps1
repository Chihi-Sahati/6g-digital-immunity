Write-Host "Rebuilding LLM Agent..."
docker compose build llm-agent
docker compose up -d llm-agent

Write-Host "Waiting 5 seconds for services to settle..."
Start-Sleep -Seconds 5

Write-Host "Copying collect_metrics.py to llm-agent container..."
docker cp evaluation/collect_metrics.py 6gdi-llm-agent:/app/collect_metrics.py

Write-Host "Collecting metrics from Kafka for 30 seconds..."
docker exec -e KAFKA_BOOTSTRAP_SERVERS="kafka:9092" 6gdi-llm-agent python /app/collect_metrics.py

Write-Host "Retrieving metrics_log.json..."
docker cp 6gdi-llm-agent:/app/metrics_log.json evaluation/metrics_log.json

Write-Host "Plotting results..."
python evaluation/plot_results.py

Write-Host "Done! The plots are saved in the evaluation folder."
