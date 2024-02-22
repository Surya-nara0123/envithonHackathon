import { useState } from 'react'
import Webcam from 'react-webcam'
import './App.css'

function App() {
  const [count, setCount] = useState(0);
  const videoConstraints = {
    width: 200,
    height: 200,
    facingMode: "environment"
}

  return (
    <>
      <Webcam videoConstraints={videoConstraints}/>
    </>
  )
}

export default App
