import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Area, AreaChart, ComposedChart
} from 'recharts';
import { 
  Activity, Database, Zap, TrendingUp, DollarSign, Clock,
  Cpu, ArrowUp, ArrowDown, MapPin, Target,
  PlayCircle, Filter, ChevronRight, Server, Radio
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_PATH = '/ws/dashboard';
const wsUrl = () => {
  const base = import.meta.env.VITE_WS_BASE || API_BASE.replace(/^http/, 'ws');
  return `${base}${WS_PATH}`;
};

// Color palette - Industrial Data Ops theme
const COLORS = {
  primary: '#00D9FF',      // Cyan accent
  secondary: '#FF4D6A',    // Coral accent  
  tertiary: '#FFB800',     // Amber warning
  success: '#00FF88',      // Neon green
  danger: '#FF3366',
  neutral: '#94A3B8',
  bg: {
    main: '#0A0E1A',
    card: '#111827',
    elevated: '#1A202E'
  },
  text: {
    primary: '#F8FAFC',
    secondary: '#94A3B8',
    muted: '#64748B'
  }
};

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  
  // Overview data
  const [kpis, setKpis] = useState(null);
  const [hourlyTrends, setHourlyTrends] = useState([]);
  const [zoneHeatmap, setZoneHeatmap] = useState([]);
  
  // AI Forecast data
  const [forecastZone, setForecastZone] = useState(161);
  const [forecastHours, setForecastHours] = useState(1);
  const [modelBundle, setModelBundle] = useState('100');
  const [bundles, setBundles] = useState(['100', '230', '237']);
  const [prediction, setPrediction] = useState(null);
  const [validationHistory, setValidationHistory] = useState([]);
  const [validationMetricsByModel, setValidationMetricsByModel] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [dataSource, setDataSource] = useState('');
  const wsRef = useRef(null);

  // System health
  const [clusterHealth, setClusterHealth] = useState(null);
  const [streamingStatus, setStreamingStatus] = useState(null);

  const applyTickPayload = useCallback((m) => {
    if (m.kpis) setKpis(m.kpis);
    if (Array.isArray(m.hourlyTrends)) setHourlyTrends(m.hourlyTrends);
    if (Array.isArray(m.zoneHeatmap)) setZoneHeatmap(m.zoneHeatmap);
    if (m.prediction !== undefined) setPrediction(m.prediction);
    if (Array.isArray(m.validationHistory)) setValidationHistory(m.validationHistory);
    if (m.prediction?.validation_metrics_by_model) {
      setValidationMetricsByModel(m.prediction.validation_metrics_by_model);
    }
    if (m.clusterHealth) setClusterHealth(m.clusterHealth);
    if (m.streamingStatus) setStreamingStatus(m.streamingStatus);
    if (m.data_source) setDataSource(m.data_source);
    if (m.registry?.bundles?.length) setBundles(m.registry.bundles);
  }, []);

  const sendWsConfig = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'config', bundle_id: modelBundle, zone_id: forecastZone }));
    }
  }, [modelBundle, forecastZone]);

  const fetchOverviewData = async () => {
    try {
      const [kpisRes, trendsRes, zonesRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/api/overview/kpis`),
        fetch(`${API_BASE}/api/overview/hourly-trends`),
        fetch(`${API_BASE}/api/overview/zone-heatmap`),
        fetch(`${API_BASE}/api/system/cluster-health`)
      ]);

      setKpis(await kpisRes.json());
      setHourlyTrends((await trendsRes.json()).data);
      setZoneHeatmap((await zonesRes.json()).data);
      setClusterHealth(await healthRes.json());
      setLoading(false);
    } catch (error) {
      console.error('Fetch error:', error);
    }
  };

  const fetchForecastData = useCallback(async () => {
    try {
      const qs = new URLSearchParams({
        bundle_id: modelBundle,
        zone_id: String(forecastZone),
      });
      const [validationRes, streamRes] = await Promise.all([
        fetch(`${API_BASE}/api/forecast/validation-history?${qs}`),
        fetch(`${API_BASE}/api/system/streaming-status`),
      ]);

      const vj = await validationRes.json();
      setValidationHistory(vj.data || []);
      setValidationMetricsByModel(vj.validation_metrics_by_model || null);
      setStreamingStatus(await streamRes.json());
    } catch (error) {
      console.error('Forecast fetch error:', error);
    }
  }, [forecastZone, modelBundle]);

  const runPrediction = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/api/forecast/predict?zone_id=${forecastZone}&hours_ahead=${forecastHours}&bundle_id=${modelBundle}`
      );
      const pred = await res.json();
      setPrediction(pred);
      if (pred?.validation_metrics_by_model) {
        setValidationMetricsByModel(pred.validation_metrics_by_model);
      }
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  useEffect(() => {
    let reconnectTimer;
    const connect = () => {
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        setWsConnected(true);
        ws.send(JSON.stringify({ type: 'config', bundle_id: modelBundle, zone_id: forecastZone }));
      };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === 'hello') {
            if (Array.isArray(m.bundles) && m.bundles.length) setBundles(m.bundles);
            setLoading(false);
            return;
          }
          if (m.type === 'tick') {
            applyTickPayload(m);
            setLoading(false);
          }
        } catch (e) {
          console.error('WS parse error', e);
        }
      };
      ws.onerror = () => setWsConnected(false);
      ws.onclose = () => {
        setWsConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      };
    };
    connect();
    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [applyTickPayload]);

  useEffect(() => {
    sendWsConfig();
  }, [modelBundle, forecastZone, sendWsConfig]);

  useEffect(() => {
    if (activeTab === 'forecast') fetchForecastData();
  }, [activeTab, forecastZone, modelBundle, fetchForecastData]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!wsConnected) {
        fetchOverviewData();
        if (activeTab === 'forecast') fetchForecastData();
      }
    }, 8000);
    return () => clearInterval(interval);
  }, [activeTab, wsConnected, fetchForecastData]);

  useEffect(() => {
    if (activeTab === 'forecast' && !prediction && !wsConnected) {
      runPrediction();
    }
  }, [activeTab, prediction, wsConnected, modelBundle, forecastZone, forecastHours]);

  useEffect(() => {
    const t = setTimeout(() => setLoading(false), 4000);
    return () => clearTimeout(t);
  }, []);

  const formatNumber = (num) => {
    if (!num) return '0';
    return new Intl.NumberFormat('en-US').format(Math.round(num));
  };

  const formatCurrency = (num) => {
    if (!num) return '$0';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(num);
  };

  const perModelMetrics = useMemo(() => {
    if (
      prediction &&
      !prediction.error &&
      Number(prediction.zone_id) === Number(forecastZone) &&
      String(prediction.bundle_id) === String(modelBundle) &&
      prediction.validation_metrics_by_model
    ) {
      return prediction.validation_metrics_by_model;
    }
    return validationMetricsByModel;
  }, [prediction, forecastZone, modelBundle, validationMetricsByModel]);

  if (loading) {
    return (
      <div className="h-screen bg-[#0A0E1A] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-[#00D9FF] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <div className="text-[#94A3B8] font-mono text-sm tracking-wider">
            INITIALIZING SPARK CLUSTER
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-[#0A0E1A] text-[#F8FAFC] flex overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-[#111827] border-r border-[#1E293B] flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-[#1E293B]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-[#00D9FF] to-[#0066FF] rounded-lg flex items-center justify-center">
              <Database className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-white font-bold text-lg leading-tight tracking-tight">
                NYC TLC
              </h1>
              <p className="text-[#64748B] text-xs font-mono uppercase tracking-wider">
                DATA OPS
              </p>
              <p className="text-[10px] font-mono mt-1 text-[#64748B]">
                WS {wsConnected ? <span className="text-[#00FF88]">LIVE</span> : <span className="text-[#FFB800]">…</span>}
                {dataSource ? ` · ${dataSource}` : ''}
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          <button
            onClick={() => setActiveTab('overview')}
            className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-all ${
              activeTab === 'overview'
                ? 'bg-[#1A202E] text-[#00D9FF] border border-[#00D9FF]/20'
                : 'text-[#94A3B8] hover:bg-[#1A202E] hover:text-white'
            }`}
          >
            <TrendingUp size={18} />
            <span className="font-medium text-sm">Operations View</span>
            {activeTab === 'overview' && <ChevronRight className="ml-auto" size={16} />}
          </button>

          <button
            onClick={() => setActiveTab('forecast')}
            className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-all ${
              activeTab === 'forecast'
                ? 'bg-[#1A202E] text-[#00D9FF] border border-[#00D9FF]/20'
                : 'text-[#94A3B8] hover:bg-[#1A202E] hover:text-white'
            }`}
          >
            <Target size={18} />
            <span className="font-medium text-sm">AI Forecasting</span>
            {activeTab === 'forecast' && <ChevronRight className="ml-auto" size={16} />}
          </button>
        </nav>

        {/* Cluster Status */}
        <div className="p-4 border-t border-[#1E293B]">
          <div className="bg-[#1A202E] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Server size={14} className="text-[#00FF88]" />
              <span className="text-xs font-mono text-[#94A3B8] uppercase">
                Silver ingest
              </span>
            </div>
            {clusterHealth && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-[#64748B]">Parquet files OK</span>
                  <span className="text-white font-mono">{clusterHealth.active_workers}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#64748B]">Read success</span>
                  <span className="text-[#00FF88] font-mono">{clusterHealth.cpu_utilization_pct}%</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeTab === 'overview' ? (
          <OverviewTab 
            kpis={kpis}
            hourlyTrends={hourlyTrends}
            zoneHeatmap={zoneHeatmap}
            clusterHealth={clusterHealth}
            formatNumber={formatNumber}
            formatCurrency={formatCurrency}
          />
        ) : (
          <ForecastTab
            forecastZone={forecastZone}
            setForecastZone={setForecastZone}
            forecastHours={forecastHours}
            setForecastHours={setForecastHours}
            modelBundle={modelBundle}
            setModelBundle={setModelBundle}
            bundles={bundles}
            prediction={prediction}
            validationHistory={validationHistory}
            perModelMetrics={perModelMetrics}
            streamingStatus={streamingStatus}
            runPrediction={runPrediction}
            formatNumber={formatNumber}
            formatCurrency={formatCurrency}
          />
        )}
      </div>
    </div>
  );
}

function OverviewTab({ kpis, hourlyTrends, zoneHeatmap, clusterHealth, formatNumber, formatCurrency }) {
  const topZones = [...zoneHeatmap].sort((a, b) => b.trip_count - a.trip_count).slice(0, 15);

  return (
    <div className="flex-1 overflow-auto">
      {/* Header */}
      <div className="bg-[#111827] border-b border-[#1E293B] px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              Operations Dashboard
            </h2>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-[#1A202E] rounded-lg border border-[#00FF88]/20">
            <Radio className="text-[#00FF88] animate-pulse" size={16} />
            <span className="text-[#00FF88] text-sm font-mono">LIVE</span>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="px-8 py-6 grid grid-cols-4 gap-6">
        <MetricCard
          title="Trip Volume"
          value={formatNumber(kpis?.total_trips)}
          change={kpis?.total_trips_change}
          changeCaption="12–23h vs 0–11h"
          icon={TrendingUp}
          color={COLORS.primary}
          suffix="rides"
        />
        <MetricCard
          title="Total Revenue"
          value={formatCurrency(kpis?.total_revenue)}
          change={kpis?.total_revenue_change}
          changeCaption="12–23h vs 0–11h"
          icon={DollarSign}
          color={COLORS.success}
        />
        <MetricCard
          title="Avg Duration"
          value={kpis?.avg_trip_duration}
          icon={Clock}
          color={COLORS.tertiary}
          suffix="min"
          small
        />
        <MetricCard
          title="Avg Fare"
          value={formatCurrency(kpis?.avg_fare)}
          icon={DollarSign}
          color={COLORS.secondary}
          small
        />
      </div>

      {/* Charts Row */}
      <div className="px-8 pb-8 grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 340px)' }}>
        {/* Hourly Trends */}
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6 flex flex-col">
          <div className="mb-4">
            <h3 className="text-lg font-semibold">Demand & Revenue by Hour</h3>
          </div>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourlyTrends}>
                <defs>
                  <linearGradient id="tripGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={COLORS.primary} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.secondary} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={COLORS.secondary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis 
                  dataKey="hour_label" 
                  stroke="#64748B"
                  style={{ fontSize: 10, fontFamily: 'monospace' }}
                />
                <YAxis 
                  yAxisId="trips"
                  stroke={COLORS.primary}
                  style={{ fontSize: 10, fontFamily: 'monospace' }}
                />
                <YAxis 
                  yAxisId="revenue"
                  orientation="right"
                  stroke={COLORS.secondary}
                  style={{ fontSize: 10, fontFamily: 'monospace' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1A202E', 
                    border: '1px solid #334155',
                    borderRadius: 8,
                    fontSize: 12
                  }}
                />
                <Area 
                  yAxisId="trips"
                  type="monotone"
                  dataKey="trip_count"
                  stroke={COLORS.primary}
                  strokeWidth={2}
                  fill="url(#tripGradient)"
                  name="Trips"
                />
                <Area 
                  yAxisId="revenue"
                  type="monotone"
                  dataKey="revenue"
                  stroke={COLORS.secondary}
                  strokeWidth={2}
                  fill="url(#revenueGradient)"
                  name="Revenue ($)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Zone Heatmap */}
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6 flex flex-col">
          <div className="mb-4">
            <h3 className="text-lg font-semibold">High-Demand Zones</h3>
          </div>
          <div className="flex-1 overflow-auto">
            <div className="space-y-2">
              {topZones.map((zone, idx) => {
                const maxTrips = topZones[0].trip_count;
                const percentage = (zone.trip_count / maxTrips) * 100;
                
                return (
                  <div key={zone.zone_id} className="group">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-[#64748B] text-xs font-mono w-6 flex-shrink-0">
                          #{idx + 1}
                        </span>
                        <MapPin size={12} className="text-[#00D9FF] flex-shrink-0" />
                        <span className="text-sm truncate">{zone.zone_name}</span>
                      </div>
                      <span className="text-white font-mono text-sm ml-2 flex-shrink-0">
                        {formatNumber(zone.trip_count)}
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#1A202E] rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-[#00D9FF] to-[#0066FF] transition-all duration-500"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ForecastTab({ 
  forecastZone, setForecastZone, forecastHours, setForecastHours,
  modelBundle, setModelBundle, bundles,
  prediction, validationHistory, perModelMetrics, streamingStatus,
  runPrediction, formatNumber, formatCurrency,
}) {
  const zones = [
    { id: 161, name: "Midtown Center" },
    { id: 100, name: "Garment District" },
    { id: 230, name: "Times Square" },
    { id: 237, name: "Upper East Side South" },
    { id: 162, name: "Midtown East" },
    { id: 236, name: "Upper East Side" },
    { id: 79, name: "East Village" },
    { id: 113, name: "Greenwich Village" }
  ];

  const modelRows = perModelMetrics
    ? ['ensemble', 'lstm', 'xgboost', 'random_forest'].filter((k) => perModelMetrics[k])
    : [];

  return (
    <div className="flex-1 overflow-auto">
      <div className="bg-[#111827] border-b border-[#1E293B] px-8 py-6">
        <h2 className="text-2xl font-bold tracking-tight">
          AI Demand Forecasting
        </h2>
      </div>

      {/* Control Panel */}
      <div className="px-8 py-6 bg-[#111827] border-b border-[#1E293B]">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-[#64748B]" />
            <label className="text-sm text-[#94A3B8] font-mono uppercase tracking-wider">
              Zone
            </label>
            <select
              value={forecastZone}
              onChange={(e) => setForecastZone(Number(e.target.value))}
              className="bg-[#1A202E] border border-[#334155] text-white px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-[#00D9FF]"
            >
              {zones.map(z => (
                <option key={z.id} value={z.id}>{z.name}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-[#64748B]" />
            <label className="text-sm text-[#94A3B8] font-mono uppercase tracking-wider">
              Model bundle
            </label>
            <select
              value={modelBundle}
              onChange={(e) => setModelBundle(e.target.value)}
              className="bg-[#1A202E] border border-[#334155] text-white px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-[#00D9FF] max-w-[200px]"
            >
              {bundles.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <Clock size={16} className="text-[#64748B]" />
            <label className="text-sm text-[#94A3B8] font-mono uppercase tracking-wider">
              Horizon
            </label>
            <select
              value={forecastHours}
              onChange={(e) => setForecastHours(Number(e.target.value))}
              className="bg-[#1A202E] border border-[#334155] text-white px-3 py-2 rounded-lg text-sm focus:outline-none focus:border-[#00D9FF]"
            >
              <option value={1}>1 Hour</option>
              <option value={2}>2 Hours</option>
              <option value={3}>3 Hours</option>
              <option value={6}>6 Hours</option>
            </select>
          </div>

          <button
            onClick={runPrediction}
            className="ml-auto bg-gradient-to-r from-[#00D9FF] to-[#0066FF] hover:from-[#00B8DD] hover:to-[#0055DD] text-white px-6 py-2.5 rounded-lg font-semibold text-sm flex items-center gap-2 transition-all shadow-lg shadow-[#00D9FF]/20"
          >
            <PlayCircle size={18} />
            Execute Inference
          </button>
        </div>
      </div>

      {/* Results Grid */}
      <div className="px-8 py-6 grid grid-cols-3 gap-6">
        {/* Prediction Result */}
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6">
          <h3 className="text-sm font-mono text-[#64748B] uppercase tracking-wider mb-4">
            Ensemble Prediction
          </h3>
          {prediction?.error ? (
            <div className="text-sm text-[#FFB800] font-mono">{prediction.error}</div>
          ) : prediction?.predictions ? (
            <>
              <div className="mb-6">
                <div className="text-5xl font-bold bg-gradient-to-r from-[#00D9FF] to-[#00FF88] bg-clip-text text-transparent mb-2">
                  {formatNumber(prediction.predictions.ensemble)}
                </div>
                <div className="text-[#94A3B8] text-sm">expected trips · bundle {prediction.bundle_id || modelBundle}</div>
              </div>
              
              <div className="space-y-3 pt-4 border-t border-[#1E293B]">
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">LSTM Model</span>
                  <span className="text-white font-mono">{formatNumber(prediction.predictions.lstm)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">XGBoost Model</span>
                  <span className="text-white font-mono">{formatNumber(prediction.predictions.xgboost)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">Random Forest</span>
                  <span className="text-white font-mono">{formatNumber(prediction.predictions.random_forest)}</span>
                </div>
                {prediction.revenue_estimates && (
                  <div className="pt-3 border-t border-[#1E293B] space-y-2">
                    <div className="text-[10px] font-mono text-[#64748B] uppercase tracking-wider">Revenue</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#64748B]">Ensemble</span>
                      <span className="text-[#FFB800] font-mono">{formatCurrency(prediction.revenue_estimates.ensemble)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-[#64748B]">LSTM / XGB / RF</span>
                      <span className="text-white font-mono text-right leading-tight">
                        {formatCurrency(prediction.revenue_estimates.lstm)} · {formatCurrency(prediction.revenue_estimates.xgboost)} · {formatCurrency(prediction.revenue_estimates.random_forest)}
                      </span>
                    </div>
                    {prediction.revenue_estimates.avg_revenue_per_trip_window != null && (
                      <div className="text-[10px] text-[#64748B] font-mono">
                        Avg $/trip: {prediction.revenue_estimates.avg_revenue_per_trip_window}
                      </div>
                    )}
                  </div>
                )}
                <div className="flex justify-between text-sm pt-3 border-t border-[#1E293B]">
                  <span className="text-[#64748B]">Confidence</span>
                  <span className="text-[#00FF88] font-mono">{((prediction.confidence_score || 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">Inference Time</span>
                  <span className="text-[#00D9FF] font-mono">{(prediction.inference_time_ms || 0).toFixed(0)}ms</span>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-8 text-[#64748B]">
              <Activity className="mx-auto mb-2 animate-pulse" size={32} />
              <div className="text-sm">Execute inference to see results</div>
            </div>
          )}
        </div>

        {/* Model Metrics */}
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6">
          <h3 className="text-sm font-mono text-[#64748B] uppercase tracking-wider mb-4">
            Model Performance
          </h3>
          {prediction?.predictions || modelRows.length > 0 ? (
            <div className="space-y-4">
              {prediction?.predictions && (
                <>
                  <MetricDisplay
                    label="RMSE"
                    value={(prediction.model_metrics?.rmse ?? 0).toFixed(2)}
                    unit="trips"
                    color={COLORS.secondary}
                  />
                  <MetricDisplay
                    label="MAE"
                    value={(prediction.model_metrics?.mae ?? 0).toFixed(2)}
                    unit="trips"
                    color={COLORS.tertiary}
                  />
                  <MetricDisplay
                    label="R² Score"
                    value={(prediction.model_metrics?.r2_score ?? 0).toFixed(3)}
                    color={COLORS.success}
                    isPercentage={false}
                  />
                </>
              )}
              {modelRows.length > 0 && (
                <div className={prediction?.predictions ? 'pt-3 border-t border-[#1E293B]' : ''}>
                  <div className="text-[10px] font-mono text-[#64748B] uppercase tracking-wider mb-2">
                    By model
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs font-mono">
                      <thead>
                        <tr className="text-[#64748B] text-left">
                          <th className="pb-1 pr-2">Model</th>
                          <th className="pb-1 pr-2">RMSE</th>
                          <th className="pb-1 pr-2">MAE</th>
                          <th className="pb-1">R²</th>
                        </tr>
                      </thead>
                      <tbody>
                        {modelRows.map((key) => {
                          const m = perModelMetrics[key];
                          const label = key === 'random_forest' ? 'RF' : key.charAt(0).toUpperCase() + key.slice(1);
                          return (
                            <tr key={key} className="text-white border-t border-[#1E293B]/80">
                              <td className="py-1 pr-2 text-[#94A3B8]">{label}</td>
                              <td className="py-1 pr-2">{m.rmse?.toFixed?.(2) ?? m.rmse}</td>
                              <td className="py-1 pr-2">{m.mae?.toFixed?.(2) ?? m.mae}</td>
                              <td className="py-1">{m.r2_score?.toFixed?.(3) ?? m.r2_score}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {prediction?.predictions && (
                <div className="pt-4 border-t border-[#1E293B]">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-[#64748B]">Inference (CPU)</span>
                    <span className="text-white font-mono">{prediction.executor_nodes}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-[#64748B]">Target Zone</span>
                    <span className="text-white font-mono text-xs">{prediction.zone_name}</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-[#64748B]">
              <Zap className="mx-auto mb-2" size={32} />
              <div className="text-sm">Metrics will appear after validation data loads</div>
            </div>
          )}
        </div>

        {/* Streaming Status */}
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6">
          <h3 className="text-sm font-mono text-[#64748B] uppercase tracking-wider mb-4">
            Parquet ingest
          </h3>
          {streamingStatus && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 px-3 py-2 bg-[#1A202E] rounded-lg">
                <div className="w-2 h-2 bg-[#00FF88] rounded-full animate-pulse"></div>
                <span className="text-sm font-mono text-[#00FF88]">{streamingStatus.status}</span>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">Rows/sec (ingest)</span>
                  <span className="text-white font-mono">{formatNumber(streamingStatus.messages_per_second)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">Ingest wall time</span>
                  <span className="text-[#00D9FF] font-mono">{streamingStatus.processing_latency_ms}ms</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">Files skipped</span>
                  <span className="text-[#FFB800] font-mono">{formatNumber(streamingStatus.offset_lag)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#64748B]">WS tick interval</span>
                  <span className="text-white font-mono">{streamingStatus.batch_interval_seconds}s</span>
                </div>
              </div>

              <div className="pt-4 border-t border-[#1E293B]">
                {streamingStatus.kafka_topics.map(topic => (
                  <div key={topic} className="text-xs font-mono text-white bg-[#1A202E] px-2 py-1 rounded mb-1">
                    {topic}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Validation Chart */}
      <div className="px-8 pb-8">
        <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6">
          <div className="mb-4">
            <h3 className="text-lg font-semibold">Historical Validation</h3>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={validationHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#64748B"
                style={{ fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                yAxisId="demand"
                stroke="#64748B"
                style={{ fontSize: 10, fontFamily: 'monospace' }}
                label={{ value: 'Trips', angle: -90, position: 'insideLeft', fill: '#64748B', fontSize: 10 }}
              />
              <YAxis
                yAxisId="revenue"
                orientation="right"
                stroke={COLORS.tertiary}
                style={{ fontSize: 10, fontFamily: 'monospace' }}
                label={{ value: 'Revenue $', angle: 90, position: 'insideRight', fill: '#64748B', fontSize: 10 }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1A202E', 
                  border: '1px solid #334155',
                  borderRadius: 8,
                  fontSize: 12
                }}
              />
              <Legend />
              <Line 
                yAxisId="demand"
                type="monotone"
                dataKey="actual"
                stroke={COLORS.primary}
                strokeWidth={3}
                dot={false}
                name="Actual demand"
              />
              <Line 
                yAxisId="demand"
                type="monotone"
                dataKey="predicted"
                stroke={COLORS.secondary}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Pred. demand (ensemble)"
              />
              <Line
                yAxisId="revenue"
                type="monotone"
                dataKey="actual_revenue"
                stroke={COLORS.success}
                strokeWidth={2}
                dot={false}
                name="Actual revenue"
              />
              <Line
                yAxisId="revenue"
                type="monotone"
                dataKey="predicted_revenue"
                stroke={COLORS.tertiary}
                strokeWidth={2}
                strokeDasharray="3 3"
                dot={false}
                name="Pred. revenue (ensemble)"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, change, changeCaption, icon: Icon, color, suffix, small }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1E293B] p-6 hover:border-[#334155] transition-all group">
      <div className="flex items-start justify-between mb-3">
        <div 
          className="w-10 h-10 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon size={20} style={{ color }} />
        </div>
        {change !== undefined && change !== null && (
          <div className="text-right max-w-[140px]">
            <div className={`flex items-center justify-end gap-1 ${change >= 0 ? 'text-[#00FF88]' : 'text-[#FF3366]'}`}>
              {change >= 0 ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
              <span className="text-xs font-mono font-semibold">{Math.abs(change)}%</span>
            </div>
            {changeCaption && (
              <div className="text-[10px] text-[#64748B] font-mono mt-0.5 leading-tight">{changeCaption}</div>
            )}
          </div>
        )}
      </div>
      <div className="text-[#64748B] text-xs font-mono uppercase tracking-wider mb-2">
        {title}
      </div>
      <div className="flex items-baseline gap-1">
        <div className={`font-bold text-white ${small ? 'text-2xl' : 'text-3xl'}`}>
          {value}
        </div>
        {suffix && <span className="text-[#64748B] text-sm ml-1">{suffix}</span>}
      </div>
    </div>
  );
}

function MetricDisplay({ label, value, unit, color, isPercentage = false }) {
  return (
    <div>
      <div className="text-xs text-[#64748B] mb-2 font-mono uppercase tracking-wider">{label}</div>
      <div className="flex items-baseline gap-2">
        <div 
          className="text-3xl font-bold"
          style={{ color }}
        >
          {value}
        </div>
        {unit && <span className="text-[#64748B] text-sm">{unit}</span>}
      </div>
    </div>
  );
}

export default App;