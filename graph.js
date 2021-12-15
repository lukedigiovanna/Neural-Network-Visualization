
/**
 * This script is responsible for being able to generate graphs
 */

var canvas = d3.select('body')
    .append('svg')
    .style('background', 'LightGray')
    .attr('height', 150)
    .attr('width', 150);

canvas.selectAll('rect')
    .data([10, 25, 39, 49])
    .enter()
    .append('rect')
    .attr('y', 0)
    .attr('x', function(datum, index) {
        return index * 20;
    })
    .attr('width', 15)
    .attr('height', function(datum) {
        return datum;
    })
    .style('fill', 'red')
