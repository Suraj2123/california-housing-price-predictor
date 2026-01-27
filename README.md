# California Housing Price Predictor

Production-style machine learning regression service demonstrating end-to-end model training, evaluation, and deployment with a live interactive demo.

This project treats **evaluation and uncertainty as first-class concerns**, returning both predictions and a held-out test RMSE to communicate typical error — mirroring real-world ML engineering practices.

---

## Live Demo

- **Web App:** https://california-housing-price-predictor-m8ks.onrender.com/
- **API Docs (Swagger):** https://california-housing-price-predictor-m8ks.onrender.com/docs
- **Model Info:** https://california-housing-price-predictor-m8ks.onrender.com/model-info

### Example Prediction

```bash
curl -X POST "https://california-housing-price-predictor-m8ks.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"MedInc":5,"HouseAge":25,"AveRooms":5.5,"AveBedrms":1,"Population":1200,"AveOccup":2.8,"Latitude":34.05,"Longitude":-118.25}'
