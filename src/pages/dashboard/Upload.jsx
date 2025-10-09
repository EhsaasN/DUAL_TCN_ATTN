import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { setLoading, setToast } from '../../state/uiSlice';
import { api } from '../../api/api';

export default function Upload() {
  const [file, setFile] = useState(null);
  const dispatch = useDispatch();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      dispatch(setToast({ type: 'error', message: 'Please select a CSV file.' }));
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    dispatch(setLoading(true));
    try {
      await api.uploadFile(formData);
      dispatch(setToast({ type: 'success', message: 'File uploaded successfully!' }));
    } catch (err) {
      dispatch(setToast({ type: 'error', message: 'Upload failed.' }));
    } finally {
      dispatch(setLoading(false));
    }
  };

  return (
    <div className="max-w-lg mx-auto bg-card p-8 rounded-lg shadow-lg border border-border mt-8">
      <h2 className="text-2xl font-bold mb-6 text-center">Upload Data</h2>
      <form onSubmit={handleUpload} className="flex flex-col gap-4">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="w-full rounded bg-surface border border-border text-gray-300 px-3 py-2"
        />
        <button type="submit" className="py-2 rounded bg-primary hover:bg-accent font-bold">Upload</button>
      </form>
    </div>
  );
}
