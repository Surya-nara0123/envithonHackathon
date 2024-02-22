import { useState } from 'react'
import Webcam from 'react-webcam'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Webcam height={600} width={600}/>
    </>
  )
}

export default App
