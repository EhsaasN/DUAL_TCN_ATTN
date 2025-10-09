import React from 'react';

export default function Spinner() {
  return (
    <div className="flex justify-center items-center min-h-screen w-full bg-bgDark">
      <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-primary border-solid"></div>
    </div>
  );
}
