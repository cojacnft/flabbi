import React, { useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tooltip,
  IconButton,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  SaveAlt as SaveIcon,
} from '@mui/icons-material';
import * as d3 from 'd3';
import { formatUSD, formatEther, shortenAddress } from '../../utils/format';

interface Node {
  id: string;
  type: 'token' | 'pool';
  address: string;
  symbol?: string;
  liquidity?: string;
}

interface Link {
  source: string;
  target: string;
  value: string;
  type: 'in' | 'out';
  dex?: string;
}

interface PathVisualizerProps {
  nodes: Node[];
  links: Link[];
  width?: number;
  height?: number;
  loading?: boolean;
  error?: string;
  onRefresh?: () => void;
}

const PathVisualizer: React.FC<PathVisualizerProps> = ({
  nodes,
  links,
  width = 800,
  height = 600,
  loading = false,
  error = null,
  onRefresh,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!nodes.length || !links.length) return;
    drawGraph();
  }, [nodes, links, width, height]);

  const drawGraph = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const tooltip = d3.select(tooltipRef.current);

    // Create force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force("link", d3.forceLink(links).id((d: any) => d.id))
      .force("charge", d3.forceManyBody().strength(-1000))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(60));

    // Create arrow markers
    svg.append("defs").selectAll("marker")
      .data(["in", "out"])
      .enter().append("marker")
      .attr("id", d => `arrow-${d}`)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("fill", d => d === "in" ? "#4caf50" : "#f44336")
      .attr("d", "M0,-5L10,0L0,5");

    // Create links
    const link = svg.append("g")
      .selectAll("path")
      .data(links)
      .enter().append("path")
      .attr("stroke", d => d.type === "in" ? "#4caf50" : "#f44336")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .attr("marker-end", d => `url(#arrow-${d.type})`)
      .on("mouseover", (event, d) => {
        tooltip
          .style("visibility", "visible")
          .html(`
            <div>
              <strong>${d.dex || 'Transfer'}</strong><br/>
              Value: ${formatUSD(d.value)}<br/>
              ${d.type === 'in' ? 'Inflow' : 'Outflow'}
            </div>
          `);
      })
      .on("mousemove", (event) => {
        tooltip
          .style("top", (event.pageY - 10) + "px")
          .style("left", (event.pageX + 10) + "px");
      })
      .on("mouseout", () => {
        tooltip.style("visibility", "hidden");
      });

    // Create nodes
    const node = svg.append("g")
      .selectAll("g")
      .data(nodes)
      .enter().append("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add circles for nodes
    node.append("circle")
      .attr("r", d => d.type === "token" ? 30 : 20)
      .attr("fill", d => d.type === "token" ? "#1976d2" : "#9c27b0")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);

    // Add labels
    node.append("text")
      .text(d => d.symbol || shortenAddress(d.address))
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .attr("fill", "#fff")
      .style("font-size", "12px");

    // Add tooltips
    node.on("mouseover", (event, d) => {
      tooltip
        .style("visibility", "visible")
        .html(`
          <div>
            <strong>${d.type === 'token' ? 'Token' : 'Pool'}</strong><br/>
            Address: ${d.address}<br/>
            ${d.liquidity ? `Liquidity: ${formatUSD(d.liquidity)}` : ''}
          </div>
        `);
    })
    .on("mousemove", (event) => {
      tooltip
        .style("top", (event.pageY - 10) + "px")
        .style("left", (event.pageX + 10) + "px");
    })
    .on("mouseout", () => {
      tooltip.style("visibility", "hidden");
    });

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link.attr("d", (d: any) => {
        const dx = d.target.x - d.source.x;
        const dy = d.target.y - d.source.y;
        const dr = Math.sqrt(dx * dx + dy * dy);
        return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
      });

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };

  const handleSave = () => {
    const svg = svgRef.current;
    if (!svg) return;

    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svg);
    const blob = new Blob([source], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'arbitrage-path.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Paper sx={{ p: 2, position: 'relative' }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">
          Arbitrage Path Visualization
        </Typography>
        <Box>
          {onRefresh && (
            <Tooltip title="Refresh">
              <IconButton onClick={onRefresh} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Save as SVG">
            <IconButton onClick={handleSave}>
              <SaveIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ position: 'relative', width, height }}>
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.5)',
              zIndex: 1,
            }}
          >
            <CircularProgress />
          </Box>
        )}
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ overflow: 'visible' }}
        />
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            visibility: 'hidden',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: '#fff',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '12px',
            zIndex: 2,
          }}
        />
      </Box>
    </Paper>
  );
};

export default PathVisualizer;