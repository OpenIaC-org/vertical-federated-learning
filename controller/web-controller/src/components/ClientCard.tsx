import { Card, CardContent } from "@mui/material";
import { makeStyles } from "@mui/styles";
import { FC } from "react";
import Client from "../types/client.model";

const useStyles = makeStyles({
  wrapper: {
    fontSize: "20px",
    display: "flex",
    flexDirection: "row",
    flexWrap: "wrap",
    gap: "15px",
    justifyContent: "space-around",
    "& p": {
      margin: "5px",
    },
  },
});

const ClientCard: FC<{ client: Client }> = ({ client }) => {
  const classes = useStyles();
  return (
    <Card style={{ width: "500px" }}>
      <CardContent>
        <h3>{client.name}</h3>
        <div className={classes.wrapper}>
          <div>
            <p>Data size</p>
            <p>{client.data.size} bytes</p>
          </div>
          <div>
            <p>Number of {client.data.type}s</p>
            <p>{client.data.count} bytes</p>
          </div>
          {client.data.resolution && (
            <div>
              <p>Image Resolution</p>
              <p>{client.data.resolution}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ClientCard;
