export default interface Client {
  type: string;
  name: string;
  address: string;
  data: {
    type: string;
    size: number;
    resolution?: string;
    count: number;
  };
}
