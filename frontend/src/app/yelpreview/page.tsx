"use client"
import { useState } from "react"
import { ClassifierResults } from "@/interfaces/nlpresult ";



export default function HuggingFace() {
    const [text, setText] = useState("")
    const [result, setResult] = useState<ClassifierResults>([]);

  const analyze = async () => {
    const formData = new FormData()
    formData.append("text", text)

    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/yelp`, {
      method: "POST",
      body: formData,
    })

    const data = await res.json()
    setResult(data)
  }
    return (
        <main className="p-4">
        <h1 className="text-2xl font-bold mb-4">Review Classifier</h1>
        <textarea
          className="w-full p-2 border rounded mb-4"
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your text here..."
        />
        <button onClick={analyze} className="bg-blue-600 text-white px-4 py-2 rounded">
          Classify
        </button>
        {result && result.length > 0 && (<>
          <div className="mt-4">
            <h2 className="font-semibold mt-2">Label:</h2>
            <p> {'Label - ' + result[0]?.label}</p>
                </div>
          <div className="mt-4">
            <h2 className="font-semibold mt-2">Score:</h2>
            <p> {'Score - ' + result[0]?.score}</p>
                </div>
        </>
                
        )}
      </main>
    )
}