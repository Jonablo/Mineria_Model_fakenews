const express = require("express");
const axios = require("axios");
const app = express();

app.set("view engine", "ejs");
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.render("index", { result: null });
});

app.post("/predict", async (req, res) => {
  const text = req.body.text;
  try {
    const { data } = await axios.post(
      "http://localhost:8000/predict",
      { text },
      { headers: { "Content-Type": "application/json" } }
    );
    // data = { prediction: 0|1, probabilities: [p_true,p_fake] }
    res.render("index", { result: data });
  } catch (err) {
    res.render("index", { result: { error: "Error al predecir" } });
  }
});

app.listen(3000, () => console.log("UI running on http://localhost:3000"));