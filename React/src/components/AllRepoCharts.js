import React, { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import AppBar from "@mui/material/AppBar";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import BarCharts from "./BarCharts";
import Loader from "./Loader";
import LineCharts from "./LineCharts";
import BarChartsForIssues from "./BarchartsForIssues";
import StackedBarChart from "./StackedBarchart";

export default function AllRepoCharts() {
    const [loading, setLoading] = useState(true);
    const [repositories, setRepositories] = useState({});
    const [starsMap, setStarsMap] = useState({});
    const [forksMap, setForksMap] = useState({});
    const [issuesMap, setIssuesMap] = useState({});
    const [createdIssuesMap, setCreatedIssuesMap] = useState({});
    const [closedIssuesMap, setClosedIssuesMap] = useState({});
    const [createdClosedIssuesMap, setCreatedClosedIssuesMap] = useState({});
    const repositoriesToFetch = [

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

    React.useEffect(() => {
        setLoading(true);
        // Extract the name of each repository
        const repositoryNames = repositoriesToFetch.map((repo) => repo.key);
        setRepositories(repositoryNames);

        const requestOptions = {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ repository: repositoryNames }),
        };
        fetch("/api/github/fetch", requestOptions)
            .then((res) => res.json())
            .then(
                (response) => {
                    response.forEach(result => {
                        var createdIssuesArr = []
                        var closedIssuesArr = []
                        starsMap[result['repositoryName']] = result['numberOfStars']
                        forksMap[result['repositoryName']] = result['numberOfForks']
                        issuesMap[result['repositoryName']] = result['numberOfIssues'][0] + result['numberOfIssues'][1]
                        createdClosedIssuesMap[result['repositoryName']] = result['numberOfIssues']
                        loadCreatedIssuesMap(result, createdIssuesArr);
                        loadClosedIssuesMap(result, closedIssuesArr);
                    })
                    setStarsMap(starsMap);
                    setForksMap(forksMap);
                    setIssuesMap(issuesMap);
                    setCreatedClosedIssuesMap(createdClosedIssuesMap);
                    setCreatedIssuesMap(createdIssuesMap);
                    setClosedIssuesMap(closedIssuesMap);
                    setLoading(false);
                },
                (error) => {
                    console.log(error);
                    setLoading(false);
                }
            );
    }, []);

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
                        Barcharts and Linecharts
                    </Typography>
                </Toolbar>
            </AppBar>

            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
                <Toolbar />
                {loading ? (
                    <Loader />
                ) : (
                    <div>
                        <div>
                            <LineCharts title={`Issues of every Repo`}
                                data={issuesMap} yAxis='Issues' flag='true' />
                            <BarChartsForIssues title={`Issues Created for every month`}
                                data={createdIssuesMap} yLabel='Issues' />
                            <BarCharts
                                title={`Stars of every Repo`}
                                data={starsMap} yLabel='Stars' flag='true'
                            />
                            <BarCharts
                                title={`Fork of every Repo`}
                                data={forksMap} yLabel='Forks' flag='true'
                            />
                            <BarChartsForIssues title={`Issues closed for every week`}
                                data={closedIssuesMap} yLabel='Issues' />
                            <StackedBarChart title = {`Created and Closed Issues`}
                            data={createdClosedIssuesMap} yLabel='Issues' repositories = {repositories} />
                        </div>
                    </div>
                )}
            </Box>
        </Box>
    );

    function loadCreatedIssuesMap(result, createdIssuesArr) {
        result['issuesCreated'].forEach(values => {
            var map = {};
            map[values[0]] = values[1];
            createdIssuesArr.push(map);
        });
        createdIssuesMap[result['repositoryName']] = createdIssuesArr;
    }

    function loadClosedIssuesMap(result, closedIssuesArr) {
        result['issuesClosed'].forEach(values => {
            var map = {};
            map[values[0]] = values[1];
            closedIssuesArr.push(map);
        });
        closedIssuesMap[result['repositoryName']] = closedIssuesArr;
    }
}

