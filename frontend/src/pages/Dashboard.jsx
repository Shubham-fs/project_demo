import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Spinner } from '../components/ui/Spinner';
import { Users, TrendingUp, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getStudents } from '../services/api';

export const Dashboard = () => {
  const [stats, setStats] = useState({ total: 0, highConversion: 0, avgProb: 0 });
  const [chartData, setChartData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const data = await getStudents();
        
        setStats({
          total: data.length,
          highConversion: data.filter(s => s.Tier === 'HIGH').length,
          avgProb: data.length > 0 ? (data.reduce((acc, s) => acc + (s.Purchase_Probability || 0), 0) / data.length) : 0
        });

        const buckets = { '0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0 };
        data.forEach(s => {
          const p = parseFloat(s.Purchase_Probability || 0);
          if (p <= 20) buckets['0-20%']++;
          else if (p <= 40) buckets['20-40%']++;
          else if (p <= 60) buckets['40-60%']++;
          else if (p <= 80) buckets['60-80%']++;
          else buckets['80-100%']++;
        });

        setChartData(Object.keys(buckets).map(key => ({
          name: key,
          count: buckets[key]
        })));
      } catch (err) {
        console.error("Dashboard error:", err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  if (isLoading) {
    return <Spinner />;
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">Dashboard Overview</h1>
        <p className="text-slate-500 dark:text-slate-400 text-lg">At a glance view of your conversion metrics directly from the ML model.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="flex items-center gap-5 hover:-translate-y-1">
          <div className="p-4 bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-2xl shadow-inner">
            <Users className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Total Students</p>
            <p className="text-4xl font-black mt-1 text-slate-900 dark:text-white">{stats.total}</p>
          </div>
        </Card>

        <Card className="flex items-center gap-5 hover:-translate-y-1">
          <div className="p-4 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-2xl shadow-inner">
            <CheckCircle className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">High Tier Leads</p>
            <p className="text-4xl font-black mt-1 text-slate-900 dark:text-white">{stats.highConversion}</p>
          </div>
        </Card>

        <Card className="flex items-center gap-5 hover:-translate-y-1">
          <div className="p-4 bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 rounded-2xl shadow-inner">
            <TrendingUp className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Avg Probability</p>
            <p className="text-4xl font-black mt-1 text-slate-900 dark:text-white">
              {stats.avgProb.toFixed(1)}<span className="text-2xl text-slate-400 font-bold">%</span>
            </p>
          </div>
        </Card>
      </div>

      <Card className="mt-8 border-t-4 border-indigo-500 dark:border-indigo-500">
        <h3 className="text-xl font-bold tracking-tight mb-8">Purchase Probability Distribution</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{fill: '#64748b', fontSize: 13, fontWeight: 500}} axisLine={false} tickLine={false} dy={10} />
              <YAxis tick={{fill: '#64748b', fontSize: 13, fontWeight: 500}} axisLine={false} tickLine={false} dx={-10} />
              <Tooltip 
                cursor={{fill: 'rgba(99, 102, 241, 0.05)'}}
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', fontWeight: 600 }}
              />
              <Bar dataKey="count" fill="#6366f1" radius={[6, 6, 0, 0]} barSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};
