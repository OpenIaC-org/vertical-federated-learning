import { Button, TextField } from "@mui/material";
import React, { useState } from "react";
import "./App.css";
import ClientCard from "./components/ClientCard";
import ServerCard from "./components/ServerCard";
import Client from "./types/client.model";
import Server from "./types/server.model";

function App() {
  const [server, setServer] = useState<Server>();
  const [clients, setClients] = useState<Client[]>([]);
  const [ip, setIp] = useState<string>("");

  const addServerOrClient = (type: "server" | "client") => {
    if (ip === "") return;

    fetch(`http://${ip}/metadata`)
      .then((res) => res.json())
      .then((data) => {
        type === "server"
          ? setServer({ ...data, address: ip })
          : setClients([...clients, { ...data, address: ip }]);
      });
  };

  return (
    <div className="App App-header">
      <h1>Federated Learning Dashboard</h1>
      <div>
        <p>IP:PORT</p>
        <TextField
          InputProps={{
            style: { color: "#fff" },
          }}
          onChange={(e) => setIp(e.target.value)}
        />
        <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
          <Button
            variant="contained"
            onClick={() => {
              addServerOrClient("server");
            }}
          >
            Add Server
          </Button>
          <Button
            variant="contained"
            onClick={() => {
              addServerOrClient("client");
            }}
          >
            Add Client
          </Button>
        </div>
      </div>
      <div style={{ margin: "20px" }}>
        {server && <ServerCard server={server} />}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          flexWrap: "wrap",
          gap: "10px",
        }}
      >
        {clients.map((client, i) => (
          <ClientCard key={`client-${i}`} client={client} />
        ))}
      </div>
    </div>
  );
}

export default App;
