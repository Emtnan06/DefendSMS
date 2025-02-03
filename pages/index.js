import { useState } from "react";
import Image from "next/image";

export default function Home() {
    const [message, setMessage] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setPrediction(null);
        
        try {
            const response = await fetch("http://localhost:5001/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });
            
            const data = await response.json();
            setPrediction(data.prediction);
        } catch (error) {
            console.error("Error:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", textAlign: "center", backgroundColor: "#f4f4f4", padding: "20px" }}>
            {/* Navbar */}
            <nav style={{ position: "fixed", top: 0, left: 0, width: "100%", background: "#C8AFD8", padding: "15px 20px", display: "flex", alignItems: "center", justifyContent: "space-between", boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <Image src="/logo.png" alt="Logo" width={140} height={100} />
                    <h2 style={{ margin: 0, fontSize: "20px", fontWeight: "bold" }}>DefendSMS</h2>
                </div>
            </nav>
            
            <div style={{ width: "600px", background: "white", padding: "30px", borderRadius: "12px", boxShadow: "0px 8px 20px rgba(0, 0, 0, 0.1)", marginTop: "120px" }}>
                <header style={{ textAlign: "center", marginBottom: "20px" }}>
                    <h1 style={{ fontSize: "28px", fontWeight: "bold" }}> SMS Phishing Detection </h1>
                    <p style={{ fontSize: "18px", color: "#555" }}>Model CNN-LSTM with BERT </p>
                </header>
                <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "15px", alignItems: "center" }}>
                    <textarea
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Enter a message to analyze"
                        rows={4}
                        style={{ padding: "12px", borderRadius: "8px", border: "1px solid #ccc", width: "100%", fontSize: "16px" }}
                        required
                    />
                    <button type="submit" disabled={loading} style={{ padding: "14px", borderRadius: "8px", border: "none", backgroundColor: "#B38BCE", color: "white", cursor: "pointer", fontSize: "18px", fontWeight: "bold", width: "50%" }}>
                        {loading ? "Analyzing..." : "Check Message"}
                    </button>
                </form>
                {prediction !== null && (
                    <h2 style={{ marginTop: "20px", fontSize: "22px", fontWeight: "bold", textAlign: "center", color: prediction === 1 ? "red" : "green" }}>
                        {prediction === 1 ? "⚠️ This is a phishing message!" : " This message is safe "}
                    </h2>
                )}
            </div>
            <footer style={{ marginTop: "30px", fontSize: "16px", color: "#777", fontWeight: "bold" }}>
                <p>Model Designed by: <strong>Emtnan - Jana - Sadeem - Lubna</strong></p>
            </footer>
        </div>
    );
}
