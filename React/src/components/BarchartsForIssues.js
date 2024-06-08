import React from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

const BarChartsForIssues = (props) => {
    var extractedRepoData = []
    const data = Object.entries(props.data).map(([repositoryName, values]) => ({
        repositoryName,
        val: values.map((item) => {
            const [attribute, count] = Object.entries(item)[0];
            return [attribute, count]
        }),
    }));
    extractedRepoData = data;
    const config = {
        chart: {
            type: "column",
        },
        title: {
            text: props.title,
        },
        xAxis: {
            type: "category",
            categories: extractedRepoData[3].val.map((item) => item[0]),
            labels: {
                rotation: -45,
                style: {
                    fontSize: "13px",
                    fontFamily: "Verdana, sans-serif",
                },
            },
        },
        yAxis: {
            min: 0,
            title: {
                text: props.yLabel,
            },
        },
        legend: {
            enabled: false,
        },
        tooltip: {
            formatter: function () {
                return '<b>' + this.series.name + '</b><br/>' +
                    this.point.y;
            },
        },
        series: extractedRepoData.map((data) => ({
            name: data.repositoryName,
            data: data.val.map((item) => item[1]),
        }))
    };
    return (
        <div>
            <div>
                <HighchartsReact
                    highcharts={Highcharts}
                    options={config}
                ></HighchartsReact>
            </div>
        </div>
    );
};

export default BarChartsForIssues;
