var express = require("express");
var router = express.Router();
require("dotenv").config();
var plotly = require("plotly")(
  process.env.PLOTLY_USERNAME,
  process.env.PLOTLY_API_KEY
);

var data = [{ x: [0, 1, 2], y: [3, 2, 1], type: "bar" }];
var layout = { fileopt: "overwrite", filename: "simple-node-example" };

plotly.plot(data, layout, function (err, msg) {
  if (err) return console.log(err);
  console.log(msg);
});

/* GET users listing. */
router.get("/", function (req, res, next) {
  res.render("visualization");
});

module.exports = router;
