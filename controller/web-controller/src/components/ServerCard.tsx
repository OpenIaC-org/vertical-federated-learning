import { Button, Card, CardContent } from "@mui/material";
import { FC, useState } from "react";
import Server from "../types/server.model";

const ServerCard: FC<{ server: Server }> = ({ server }) => {
  const [training, setTraining] = useState(false);

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
        <h3>Server</h3>
        <Button
          disabled={training}
          variant="contained"
          onClick={handleStartTraining}
        >
          {training ? "Training..." : "Start training"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default ServerCard;
