import { Button, Card, CardContent, TextField } from "@mui/material";
import { FC, useEffect, useState } from "react";
import Server from "../types/server.model";

const ServerCard: FC<{ server: Server }> = ({ server }) => {
  const [training, setTraining] = useState(false);
  const [imageId, setImageId] = useState("");
  const [prediction, setPrediction] = useState("");
  const [label, setLabel] = useState("");

  useEffect(() => {
    fetch(`http://${server.address}/predict/${imageId}`)
      .then((res) => res.json())
      .then((res) => {
        setPrediction(`${res.prediction}`);
        setLabel(`${res.label}`);
      })
      .catch(() => {
        setPrediction("");
        setLabel("");
      });
  }, [imageId]);

  const handleStartTraining = () => {
    setTraining(true);
    fetch(`http://${server.address}/train`, {})
      .then(() => {
        setTraining(false);
      })
      .catch(() => {
        setTraining(false);
      });
  };

  return (
    <Card>
      <CardContent>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <h3>Server</h3>
          <Button
            disabled={training}
            variant="contained"
            onClick={handleStartTraining}
          >
            {training ? "Training..." : "Start training"}
          </Button>
          <TextField
            label="Enter image id to predict"
            onChange={(e) => setImageId(e.target.value)}
          />
          {prediction && label && (
            <>
              <h5>
                Predicted: {prediction}. Ground truth: {label}
              </h5>
              <img src={`http://${server.address}/image/${imageId}`} />
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ServerCard;
