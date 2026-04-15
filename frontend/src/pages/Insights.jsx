import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Spinner } from '../components/ui/Spinner';
import { getStudents } from '../services/api';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell
} from 'recharts';

export const Insights = () => {
  const [data, setData] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const students = await getStudents();
        const safeStudents = students || [];
        setData(safeStudents);
        
        if (safeStudents.length > 0) {
          const groupedData = {};
          safeStudents.forEach(s => {
            if (s.Days_Since_Login !== undefined) {
              const day = s.Days_Since_Login;
              if (!groupedData[day]) {
                groupedData[day] = { count: 0, sumProb: 0 };
              }
              groupedData[day].count += 1;
              groupedData[day].sumProb += (s.Purchase_Probability || 0);
            }
          });
          
          const avgByDay = Object.keys(groupedData)
            .map(day => ({
              Days_Since_Login: parseInt(day),
              avgProb: parseFloat((groupedData[day].sumProb / groupedData[day].count).toFixed(2))
            }))
            .sort((a, b) => a.Days_Since_Login - b.Days_Since_Login);
            
          setTrendData(avgByDay);
        }
        
      } catch (e) {
        console.error("Failed to load insights.", e);
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
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">Model Insights</h1>
        <p className="text-slate-500 dark:text-slate-400 text-lg">Deep dive into feature metrics and conversion models.</p>
      </div>

      {!data || data.length === 0 ? (
        <Card className="text-center py-20 flex flex-col items-center justify-center border-dashed border-2 bg-slate-50/50 dark:bg-slate-900/20">
          <div className="p-4 rounded-full bg-slate-100 dark:bg-slate-800 mb-4 opacity-70">
            <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-slate-700 dark:text-slate-300 mb-2">No Model Data Available</h3>
          <p className="text-slate-500 font-medium">Make sure your ML backend is running and returning students.</p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card className="border-t-4 border-indigo-500 dark:border-indigo-500 hover:shadow-md transition-shadow">
            <h3 className="text-xl font-bold mb-2">Probability vs Recency</h3>
            <p className="text-sm font-medium text-slate-500 mb-8 border-b border-slate-100 dark:border-slate-800 pb-4">Scatter plot correlation between days since last login and conversion probability.</p>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" opacity={0.6} />
                  <XAxis type="number" dataKey="Days_Since_Login" name="Days Since Login" tick={{fontSize: 13, fontWeight: 500, fill: '#64748b'}} axisLine={false} tickLine={false} dy={10} />
                  <YAxis type="number" dataKey="Purchase_Probability" name="Probability (%)" tick={{fontSize: 13, fontWeight: 500, fill: '#64748b'}} axisLine={false} tickLine={false} dx={-10} />
                  <Tooltip 
                    cursor={{strokeDasharray: '3 3', stroke: '#94a3b8'}} 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 20px -5px rgba(0, 0, 0, 0.15)', fontWeight: 600 }} 
                  />
                  <Scatter name="Students" data={data}>
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={(entry.Purchase_Probability || 0) > 50 ? '#10b981' : '#ef4444'} className="drop-shadow-sm opacity-80" />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="border-t-4 border-purple-500 dark:border-purple-500 hover:shadow-md transition-shadow">
            <h3 className="text-xl font-bold mb-2">Decay Analysis</h3>
            <p className="text-sm font-medium text-slate-500 mb-8 border-b border-slate-100 dark:border-slate-800 pb-4">Average conversion probability trend over time away from platform.</p>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="Days_Since_Login" tick={{fontSize: 13, fontWeight: 500, fill: '#64748b'}} axisLine={false} tickLine={false} dy={10} />
                  <YAxis tick={{fontSize: 13, fontWeight: 500, fill: '#64748b'}} axisLine={false} tickLine={false} dx={-10} />
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" opacity={0.6} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 20px -5px rgba(0, 0, 0, 0.15)', fontWeight: 600 }} 
                  />
                  <Area type="monotone" dataKey="avgProb" stroke="#8b5cf6" strokeWidth={4} fillOpacity={1} fill="url(#colorAvg)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};
