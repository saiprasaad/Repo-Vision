import HighchartsReact from "highcharts-react-official"
import Highcharts from "highcharts"

const LineCharts = (props) => {
    // Extracting the data for plotting
    const extractedData = Object.keys(props.data).map((key) => ({
        name: key,
        y: props.data[key],
    }));
    // Declaring the options for the graph
    const options = {
        title: {
            text: props.title,
        },
        yAxis: {
            min: 0,
            title: {
                text: props.yAxis,
            },
        },
        xAxis: {
            categories: extractedData.map(item => item.name), // Use the 'name' property for x-axis categories
        },
        series: [
            {
                name: props.title,
                data: extractedData,
                dataLabels: {
                    enabled: true,
                    rotation: -90,
                    color: "#FFFFFF", 
                    y: 20, 
                    style: {
                        fontSize: "15px",
                        fontFamily: "Arial, sans-serif",
                    },
                },
            },
        ],
        credits: {
            enabled: false
        },
    };
    return (
        <div>
            <div>
                <HighchartsReact
                    highcharts={Highcharts}
                    options={options}
                ></HighchartsReact>
            </div>
        </div>
    );
};

export default LineCharts;