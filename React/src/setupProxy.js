const { createProxyMiddleware } = require("http-proxy-middleware");

/*
This acts a proxy between the react application and the flask microservice
Everytime there is a request to /api, the setupProxy prepends the flask
microservice url mentioned in line 14
*/
module.exports = function (app) {
  app.use(
    "/api",
    createProxyMiddleware({
      // update the flask Google Cloud url
      // Do this for local
      target: "http://localhost:5000",
      // Do this for dev
      // target: "https://flask-assignment-5-new-z3y2bag5ga-uc.a.run.app",
      changeOrigin: true,
    })
  );
};