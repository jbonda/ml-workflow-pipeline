var express = require("express");
var router = express.Router();

/* GET home page. */
router.get("/", function (req, res, next) {
  res.render("index", {
    title: "CS 600",
    link: "https://github.com/jbonda/ml-workflow-pipeline",
  });
});

module.exports = router;
