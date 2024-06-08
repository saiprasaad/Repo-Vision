import React from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

const StackedBarChart = (props) => {
    const categories = ['Created', 'Closed']
    var extractedRepoData = []
    const data = Object.entries(props.data).reduce((val, [key, values], index) => {
        values.forEach((value, i) => {
            if (!val[i]) {
                val[i] = {
                    name: categories[i],
                    data: [],
                };
            }
            val[i].data.push(value);
        });
        return val;
    }, []);
    extractedRepoData = data;
    const config = {
        chart: {
            type: 'bar',
        },
        title: {
            text: props.title,
        },
        xAxis: {
            categories: props.repositories,
        },
        yAxis: {
            min: 0,
            title: {
                text: props.yAxis,
            },
        },
        legend: {
            enabled: true,
        },

        plotOptions: {
            series: {
                stacking: 'normal',
                dataLabels: {
                    enabled: true
                }
            }
        },
        series: extractedRepoData
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

export default StackedBarChart;
