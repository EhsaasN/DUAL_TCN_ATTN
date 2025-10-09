import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import Sidebar from '../../components/Sidebar';
import Spinner from '../../components/Spinner';

const Upload = lazy(() => import('./Upload'));
const AnomalyDetection = lazy(() => import('./AnomalyDetection'));
const Results = lazy(() => import('./Results'));
const Chat = lazy(() => import('./Chat'));
const Profile = lazy(() => import('./Profile'));

export default function Dashboard() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const handleLogout = () => {
    dispatch({ type: 'auth/logout' });
    navigate('/login');
  };
  return (
    <div className="flex min-h-screen bg-bgDark">
      <Sidebar onLogout={handleLogout} />
      <main className="flex-1 p-4 md:p-8 overflow-y-auto">
        <Suspense fallback={<Spinner />}>
          <Routes>
            <Route path="upload" element={<Upload />} />
            <Route path="detection" element={<AnomalyDetection />} />
            <Route path="results" element={<Results />} />
            <Route path="chat" element={<Chat />} />
            <Route path="profile" element={<Profile />} />
            <Route path="*" element={<Navigate to="upload" />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
