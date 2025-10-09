import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setToast } from '../state/uiSlice';
import { motion, AnimatePresence } from 'framer-motion';

export default function Toast() {
  const toast = useSelector((state) => state.ui.toast);
  const dispatch = useDispatch();

  return (
    <AnimatePresence>
      {toast && (
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -50, opacity: 0 }}
          transition={{ duration: 0.4 }}
          className={`fixed top-6 left-1/2 -translate-x-1/2 z-50 px-6 py-3 rounded-lg shadow-lg text-white ${toast.type === 'error' ? 'bg-red-600' : 'bg-accent'}`}
          onClick={() => dispatch(setToast(null))}
        >
          {toast.message}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
