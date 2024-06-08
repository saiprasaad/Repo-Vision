/*
Goal of React:
  1. React will retrieve GitHub created and closed issues for a given repository and will display the bar-charts 
     of same using high-charts        
  2. It will also display the images of the forecasted data for the given GitHub repository and images are being retrieved from 
     Google Cloud storage
  3. React will make a fetch api call to flask microservice.
*/

// Import required libraries
import * as React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import AppBar from "@mui/material/AppBar";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
// Import custom components
import BarCharts from "./BarCharts";
import Loader from "./Loader";
import { ListItemButton } from "@mui/material";

const drawerWidth = 240;
// List of GitHub repositories 
const repositories = [
  {
    key: "elastic/elasticsearch",
    value: "Elasticsearch",
  },
  {
    key: "milvus-io/pymilvus",
    value: "Pymilvus",
  },
  {
    key: "sebholstein/angular-google-maps",
    value: "Angular Google Maps",
  },
  {
    key: "openai/openai-python",
    value: "Open AI Python"
  },
  {
    key: "openai/openai-cookbook",
    value: "Open AI Cookbook"
  },
];

export default function Home() {
  /*
  The useState is a react hook which is special function that takes the initial 
  state as an argument and returns an array of two entries. 
  */
  /*
  setLoading is a function that sets loading to true when we trigger flask microservice
  If loading is true, we render a loader else render the Bar charts
  */
  const [loading, setLoading] = useState(true);
  /* 
  setRepository is a function that will update the user's selected repository such as Angular,
  Angular-cli, Material Design, and D3
  The repository "key" will be sent to flask microservice in a request body
  */
  const [repository, setRepository] = useState({
    key: "elastic/elasticsearch",
    value: "Elasticsearch",
  });
  /*
  
  The first element is the initial state (i.e. githubRepoData) and the second one is a function 
  (i.e. setGithubData) which is used for updating the state.

  so, setGitHub data is a function that takes the response from the flask microservice 
  and updates the value of gitHubrepo data.
  */
  const [githubRepoData, setGithubData] = useState([]);
  // Updates the repository to newly selected repository
  const eventHandler = (repo) => {
    setRepository(repo);
  };

  /* 
  Fetch the data from flask microservice on Component load and on update of new repository.
  Everytime there is a change in a repository, useEffect will get triggered, useEffect inturn will trigger 
  the flask microservice 
  */
  React.useEffect(() => {
    // set loading to true to display loader
    setLoading(true);
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Append the repository key to request body
      body: JSON.stringify({ repository: repository.key }),
    };

    /*
    Fetching the GitHub details from flask microservice
    The route "/api/github" is served by Flask/App.py in the line 53
    @app.route('/api/github', methods=['POST'])
    Which is routed by setupProxy.js to the
    microservice target: "your_flask_gcloud_url"
    */
    fetch("/api/github", requestOptions)
      .then((res) => res.json())
      .then(
        // On successful response from flask microservice
        (result) => {
          // On success set loading to false to display the contents of the resonse
          setLoading(false);
          // Set state on successfull response from the API
          setGithubData(result);
        },
        // On failure from flask microservice
        (error) => {
          // Set state on failure response from the API
          console.log(error);
          // On failure set loading to false to display the error message
          setLoading(false);
          setGithubData([]);
        }
      );
  }, [repository]);

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      {/* Application Header */}
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Timeseries Forecasting
          </Typography>
        </Toolbar>
      </AppBar>
      {/* Left drawer of the application */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: "auto" }}>
          <List>
            {/* Iterate through the repositories list */}
            {repositories.map((repo) => (
              <ListItem
                button
                key={repo.key}
                onClick={() => eventHandler(repo)}
                disabled={loading && repo.value !== repository.value}
              >
                <ListItemButton selected={repo.value === repository.value}>
                  <ListItemText primary={repo.value} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        {/* Render loader component if loading is true else render charts and images */}
        {loading ? (
          <Loader />
        ) : (
          <div>
            <div style={{ textAlign: 'center' }}>
              <h1 style={{ margin: '0 auto' }}>{repository.value}</h1>
            </div>
            <br />
            <div style={{ textAlign: 'center' }}>
              <h1 style={{ margin: '0 auto' }}>Keras / LSTM</h1>
            </div>
            <div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Created Issues using Tensorflow and
                  Keras LSTM based on past month
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Created Issues
                  </Typography>
                  {/* Render the model loss image for created issues */}
                  <img
                    src={githubRepoData?.createdAtImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Created Issues"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Created Issues
                  </Typography>
                  {/* Render the LSTM generated image for created issues*/}
                  <img
                    src={
                      githubRepoData?.createdAtImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Created Issues"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Issues Data for Created Issues
                  </Typography>
                  {/* Render the all issues data image for created issues*/}
                  <img
                    src={
                      githubRepoData?.createdAtImageUrls?.all_issues_data_image
                    }
                    alt={"All Issues Data for Created Issues"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Closed Issues using Tensorflow and
                  Keras LSTM based on past month
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Closed Issues
                  </Typography>
                  {/* Render the model loss image for closed issues  */}
                  <img
                    src={githubRepoData?.closedAtImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Closed Issues"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Closed Issues
                  </Typography>
                  {/* Render the LSTM generated image for closed issues */}
                  <img
                    src={
                      githubRepoData?.closedAtImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Closed Issues"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Issues Data for Closed Issues
                  </Typography>
                  {/* Render the all issues data image for closed issues*/}
                  <img
                    src={githubRepoData?.closedAtImageUrls?.all_issues_data_image}
                    alt={"All Issues Data for Closed Issues"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Pulls using Tensorflow and
                  Keras LSTM
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Pulls
                  </Typography>
                  <img
                    src={githubRepoData?.pullsImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Pulls"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Pulls
                  </Typography>
                  <img
                    src={
                      githubRepoData?.pullsImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Pulls"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Pulls Data
                  </Typography>
                  <img
                    src={
                      githubRepoData?.pullsImageUrls?.all_issues_data_image
                    }
                    alt={"All Pulls Data"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Commits using Tensorflow and
                  Keras LSTM
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Commits
                  </Typography>
                  <img
                    src={githubRepoData?.commitsImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Commits"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Commits
                  </Typography>
                  <img
                    src={
                      githubRepoData?.commitsImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Commits"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Commits Data
                  </Typography>
                  <img
                    src={
                      githubRepoData?.commitsImageUrls?.all_issues_data_image
                    }
                    alt={"All Commits Data"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Branches using Tensorflow and
                  Keras LSTM
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Branches
                  </Typography>
                  <img
                    src={githubRepoData?.branchesImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Branches"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Branches
                  </Typography>
                  <img
                    src={
                      githubRepoData?.branchesImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Branches"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Branches Data
                  </Typography>
                  <img
                    src={
                      githubRepoData?.branchesImageUrls?.all_issues_data_image
                    }
                    alt={"All Branches Data"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Contributors using Tensorflow and
                  Keras LSTM
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Contributors
                  </Typography>
                  <img
                    src={githubRepoData?.contributionsImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Contributors"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Contributors
                  </Typography>
                  <img
                    src={
                      githubRepoData?.contributionsImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Contributors"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Contributors Data
                  </Typography>
                  <img
                    src={
                      githubRepoData?.contributionsImageUrls?.all_issues_data_image
                    }
                    alt={"All Contributors Data"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Releases using Tensorflow and
                  Keras LSTM
                </Typography>

                <div>
                  <Typography component="h4">
                    Model Loss for Releases
                  </Typography>
                  <img
                    src={githubRepoData?.releasesImageUrls?.model_loss_image_url}
                    alt={"Model Loss for Releases"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    LSTM Generated Data for Releases
                  </Typography>
                  <img
                    src={
                      githubRepoData?.releasesImageUrls?.lstm_generated_image_url
                    }
                    alt={"LSTM Generated Data for Releases"}
                    loading={"lazy"}
                  />
                </div>
                <div>
                  <Typography component="h4">
                    All Releases Data
                  </Typography>
                  <img
                    src={
                      githubRepoData?.releasesImageUrls?.all_issues_data_image
                    }
                    alt={"All Releases Data"}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues created is {githubRepoData?.max_values?.day_max_issues_created}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues closed is {githubRepoData?.max_values?.day_max_issues_closed}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The month of the year that has maximum number of issues closed is {githubRepoData?.max_values?.month_max_issues_closed}
                </Typography>
              </div>

            </div>


            <Divider
              sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
            />
            <div>
              <div style={{ textAlign: 'center' }}>
                <h1 style={{ margin: '0 auto' }}>Stats Model</h1>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Created Issues using Stats Model
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Created Issues
                  </Typography>
                  {/* Render the all issues data image for created issues*/}
                  <img
                    src={
                      githubRepoData?.statsCreatedAt?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Closed Issues using Statsmodel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Closed Issues
                  </Typography>
                  {/* Render the all issues data image for closed issues*/}
                  <img
                    src={githubRepoData?.statsClosedAt?.stats_data_image}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Pulls using StatsModel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Pulls
                  </Typography>
                  <img
                    src={
                      githubRepoData?.statsPulls?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Commits using StatsModel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Commits
                  </Typography>
                  <img
                    src={
                      githubRepoData?.statsCommits?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Branches using StatsModel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Branches
                  </Typography>
                  <img
                    src={
                      githubRepoData?.statsBranches?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Contributors using StatsModel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Contributors
                  </Typography>
                  <img
                    src={
                      githubRepoData?.statsContributors?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Releases using StatsModel
                </Typography>
                <div>
                  <Typography component="h4">
                    StatsModel graph for Releases
                  </Typography>
                  <img
                    src={
                      githubRepoData?.statsReleases?.stats_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues created is {githubRepoData?.max_values_stat?.day_max_issues_created}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues closed is {githubRepoData?.max_values_stat?.day_max_issues_closed}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The month of the year that has maximum number of issues closed is {githubRepoData?.max_values_stat?.month_max_issues_closed}
                </Typography>
              </div>

            </div>

            <Divider
              sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
            />
            <div>
              <div style={{ textAlign: 'center' }}>
                <h1 style={{ margin: '0 auto' }}>Prophet Model</h1>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Created Issues using Prophet Model
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Created Issues
                  </Typography>
                  {/* Render the all issues data image for created issues*/}
                  <img
                    src={
                      githubRepoData?.prophetCreatedAt?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Closed Issues using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Closed Issues
                  </Typography>
                  {/* Render the all issues data image for closed issues*/}
                  <img
                    src={githubRepoData?.prophetClosedAt?.prophet_data_image}
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Pulls using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Pulls
                  </Typography>
                  <img
                    src={
                      githubRepoData?.prophetPulls?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Commits using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Commits
                  </Typography>
                  <img
                    src={
                      githubRepoData?.prophetCommits?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Branches using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Branches
                  </Typography>
                  <img
                    src={
                      githubRepoData?.prophetBranches?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Contributors using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Contributors
                  </Typography>
                  <img
                    src={
                      githubRepoData?.prophetContributors?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  Timeseries Forecasting of Releases using Prophet
                </Typography>
                <div>
                  <Typography component="h4">
                    Prophet graph for Releases
                  </Typography>
                  <img
                    src={
                      githubRepoData?.prophetReleases?.prophet_data_image
                    }
                    loading={"lazy"}
                  />
                </div>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues created is {githubRepoData?.max_values_prophet?.day_max_issues_created}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The day of the week maximum number of issues closed is {githubRepoData?.max_values_prophet?.day_max_issues_closed}
                </Typography>
              </div>
              <div>
                <Typography variant="h5" component="div" gutterBottom>
                  The month of the year that has maximum number of issues closed is {githubRepoData?.max_values_prophet?.month_max_issues_closed}
                </Typography>
              </div>

            </div>
          </div>


        )}
      </Box>
    </Box>
  );
}
