# Task 6 — Observations

1. **Online vs Batch Inference:** Online inference predicts one incoming record at a time and returns an individual result. Batch inference predicts multiple records together in one operation, which is efficient for offline processing.

2. **Streaming vs Batch Inference:** Streaming inference processes records continuously as they arrive, here simulated with a one-second delay between records. Batch inference waits until a group of records is available and processes that group together.

3. **Why FastAPI?** FastAPI provides a simple way to expose a trained model through HTTP REST endpoints. It supports request validation with Pydantic, automatically generates interactive Swagger documentation at `/docs`, and supports asynchronous web handling.

4. **Why reuse the same trained model?** Reusing the same saved pipeline keeps predictions consistent across online, batch, streaming, and API inference. It avoids unnecessary retraining and ensures that preprocessing is identical to training.

## Results to fill after running
- Test accuracy: ______
- Online prediction: ______
- Batch records generated: 20
- Streaming records processed: 10
- API `/health`: ______
- API valid `/predict`: ______
- API invalid request validation: ______
