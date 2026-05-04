# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Dict, Any, Tuple, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from src.models.data_alignment.sdf_utils import build_signed_distance_field_from_raster

class DatasetBase(Dataset[Dict[str, np.ndarray]]):
    def __init__(self, filepath: str, tensor_size: Dict[str, Tuple]) -> None:
        super().__init__()
        self.tensor_size = tensor_size
        self.filepath = filepath
        self._hf: Optional[h5py.File] = None
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            self.dataset_len = int(hf.attrs["data_len"])

    def __len__(self) -> int:
        return self.dataset_len

    def _get_hf(self) -> h5py.File:
        # Keep one HDF5 handle per worker process instead of reopening per sample.
        if self._hf is None:
            self._hf = h5py.File(self.filepath, "r", libver="latest", swmr=True)
        return self._hf

    def _close_hf(self) -> None:
        if self._hf is not None:
            try:
                self._hf.close()
            finally:
                self._hf = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hf"] = None
        return state

    def __del__(self) -> None:
        self._close_hf()


class DatasetTrain(DatasetBase):
    """
    Always train with the whole training.h5 dataset.
    limit_train_batches just for controlling the validation frequency.
    """

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # this idx calling might introduce hash collision making redundant data being loaded but it's fine for now
        # idx = np.random.randint(self.dataset_len)

        idx_key = str(idx)
        hf = self._get_hf()
        out_dict = {
            "episode_idx": idx,
            "scenario_id": hf[idx_key].attrs["scenario_id"],
            "scenario_center": hf[idx_key].attrs["scenario_center"],
            "scenario_yaw": hf[idx_key].attrs["scenario_yaw"]
        }
        for k, _size in self.tensor_size.items():
            if k in hf[idx_key]:
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
            else:
                if "/valid" in k or "/state" in k:
                    _dtype = np.bool_
                elif "/idx" in k:
                    _dtype = np.int64
                else:
                    _dtype = np.float32
                out_dict[k] = np.zeros(_size, dtype=_dtype)
        return out_dict # dont set the debug breakpoint here


class DatasetTrainRaster(DatasetBase):
    """
    Dataset class for loading raster training data that contains ground truth information
    for computing loss.
    """
    def __init__(self, filepath: str, tensor_size: Dict[str, Tuple], raster_filepath: str) -> None:
        super().__init__(filepath, tensor_size)
        self.raster_filepath = raster_filepath
        self._raster_hf: Optional[h5py.File] = None

    def _get_raster_hf(self) -> h5py.File:
        if self._raster_hf is None:
            self._raster_hf = h5py.File(self.raster_filepath, "r", libver="latest", swmr=True)
        return self._raster_hf

    def _close_raster_hf(self) -> None:
        if self._raster_hf is not None:
            try:
                self._raster_hf.close()
            finally:
                self._raster_hf = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_raster_hf"] = None
        return state

    def __del__(self) -> None:
        self._close_raster_hf()
        super().__del__()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # idx = np.random.randint(self.dataset_len)
        # print(f"[DEBUG-OVERFIT] DatasetTrain idx: {idx}")
        idx_key = str(idx)
        out_dict = {"episode_idx": idx}
        
        # Load regular data
        hf = self._get_hf()
        scenario_id = hf[idx_key].attrs["scenario_id"]
        out_dict["scenario_id"] = scenario_id
        out_dict["scenario_center"] = hf[idx_key].attrs["scenario_center"]
        out_dict["scenario_yaw"] = hf[idx_key].attrs["scenario_yaw"]
        for k, _size in self.tensor_size.items():
            if k in hf[idx_key]:
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
            else:
                if "/valid" in k or "/state" in k:
                    _dtype = np.bool_
                elif "/idx" in k:
                    _dtype = np.int64
                else:
                    _dtype = np.float32
                out_dict[k] = np.zeros(_size, dtype=_dtype)

        # Load corresponding raster data
        hf1 = self._get_raster_hf()
        scenario_key = str(scenario_id)
        # print(f"!!!!!!!!!!!!!!!!!!!scenario_key: {scenario_key}")
        assert scenario_key == str(scenario_id), f"Mismatch between keys during raster loading: {scenario_key} != {scenario_id}"
        if scenario_key in hf1:
            raster_layer = np.ascontiguousarray(hf1[scenario_key]["raster"])
            # Load Sim2 transform data
            sim2_group = hf1[scenario_key]["sim2_transform"]
            out_dict["sim2_R"] = np.ascontiguousarray(sim2_group["R"])
            out_dict["sim2_s"] = np.ascontiguousarray(sim2_group["s"])
            out_dict["sim2_t"] = np.ascontiguousarray(sim2_group["t"])
            out_dict["sdf_map"] = np.ascontiguousarray(
                build_signed_distance_field_from_raster(raster_layer, out_dict["sim2_s"])
            )
        return out_dict


class DatasetVal(DatasetBase):
    # for validation.h5 and testing.h5
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx_key = str(idx)
        hf = self._get_hf()
        out_dict = {
            "episode_idx": idx,
            "scenario_id": hf[idx_key].attrs["scenario_id"],
            "scenario_center": hf[idx_key].attrs["scenario_center"],
            "scenario_yaw": hf[idx_key].attrs["scenario_yaw"],
            "with_map": hf[idx_key].attrs["with_map"],  # some epidosdes in the testing dataset do not have map.
        }
        for k, _size in self.tensor_size.items():
            if k in hf[idx_key]:
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
            else:
                if "/valid" in k or "/state" in k:
                    _dtype = np.bool_
                elif "/idx" in k:
                    _dtype = np.int64
                else:
                    _dtype = np.float32
                out_dict[k] = np.zeros(_size, dtype=_dtype)
            if out_dict[k].shape != _size:
                assert "agent" in k
                out_dict[k] = np.ones(_size, dtype=out_dict[k].dtype)
        
        return out_dict # dont set the debug breakpoint here


class DatasetValRaster(DatasetVal):
    """
    Dataset class for loading raster validation data that contains ground truth information
    for computing loss.
    """
    def __init__(self, filepath: str, tensor_size: Dict[str, Tuple], raster_filepath: str) -> None:
        super().__init__(filepath, tensor_size)
        self.raster_filepath = raster_filepath
        self._raster_hf: Optional[h5py.File] = None

    def _get_raster_hf(self) -> h5py.File:
        if self._raster_hf is None:
            self._raster_hf = h5py.File(self.raster_filepath, "r", libver="latest", swmr=True)
        return self._raster_hf

    def _close_raster_hf(self) -> None:
        if self._raster_hf is not None:
            try:
                self._raster_hf.close()
            finally:
                self._raster_hf = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_raster_hf"] = None
        return state

    def __del__(self) -> None:
        self._close_raster_hf()
        super().__del__()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Get base validation data
        idx_key = str(idx)
        out_dict = {"episode_idx": idx}
        
        # Load regular validation data
        hf = self._get_hf()
        # Get the scenario_id first
        scenario_id = hf[idx_key].attrs["scenario_id"]
        out_dict = {
            "episode_idx": idx,
            "scenario_id": scenario_id,
            "scenario_center": hf[idx_key].attrs["scenario_center"],
            "scenario_yaw": hf[idx_key].attrs["scenario_yaw"],
            "with_map": hf[idx_key].attrs["with_map"],
        }
        
        for k, _size in self.tensor_size.items():
            if k in hf[idx_key]:
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
            else:
                if "/valid" in k or "/state" in k:
                    _dtype = np.bool_
                elif "/idx" in k:
                    _dtype = np.int64
                else:
                    _dtype = np.float32
                out_dict[k] = np.zeros(_size, dtype=_dtype)

        # Load corresponding raster validation data using scenario_id
        hf1 = self._get_raster_hf()
        scenario_key = str(scenario_id)
        assert scenario_key == str(scenario_id), f"Mismatch between keys during raster loading: {scenario_key} != {scenario_id}"

        if scenario_key in hf1:
            # out_dict["raster"] = np.ascontiguousarray(hf1[scenario_key]["raster"])
            # Load Sim2 transform data
            raster_layer = np.ascontiguousarray(hf1[scenario_key]["raster"])
            sim2_group = hf1[scenario_key]["sim2_transform"]
            out_dict["sim2_R"] = np.ascontiguousarray(sim2_group["R"])
            out_dict["sim2_s"] = np.ascontiguousarray(sim2_group["s"])
            out_dict["sim2_t"] = np.ascontiguousarray(sim2_group["t"])
            out_dict["sdf_map"] = np.ascontiguousarray(
                build_signed_distance_field_from_raster(raster_layer, out_dict["sim2_s"])
            )
        else:
            print(f"Warning: scenario_id {scenario_id} not found in raster file")
        # print(f"out_dict[scenario_id]: {out_dict['scenario_id']}")
        # print(f"out_dict[scenario_center]: {out_dict['scenario_center']}")
        # print(f"out_dict[scenario_yaw]: {out_dict['scenario_yaw']}")
        return out_dict


class DataH5av2(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        filename_train: str = "training",
        filename_val: str = "validation",
        filename_test: str = "testing",
        filename_train_raster: str = "raster_data_train",
        filename_val_raster: str = "raster_data_val",
        batch_size: int = 3,
        num_workers: int = 4,
        n_agent: int = 64,  # if not the same as h5 dataset, use dummy agents, for scalability tests.
    ) -> None:
        super().__init__()
        self.interactive_challenge = False

        self.path_train_h5 = f"{data_dir}/{filename_train}.h5"
        self.path_val_h5 = f"{data_dir}/{filename_val}.h5"
        self.path_test_h5 = f"{data_dir}/{filename_test}.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_train_raster = f"{data_dir}/{filename_train_raster}.h5" if filename_train_raster is not None else None
        self.path_val_raster = f"{data_dir}/{filename_val_raster}.h5" if filename_val_raster is not None else None

        n_step = 110
        n_step_history = 50
        n_agent_no_sim = 256
        n_pl = 1024
        n_pl_node = 20

        n_tl = 1
        n_tl_stop = 1
        self.tensor_size_train = {
            # agent states
            "agent/valid": (n_step, n_agent),  # bool,
            "agent/pos": (n_step, n_agent, 2),  # float32
            # v[1] = p[1]-p[0]. if p[1] invalid, v[1] also invalid, v[2]=v[3]
            "agent/vel": (n_step, n_agent, 2),  # float32, v_x, v_y
            "agent/spd": (n_step, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "agent/acc": (n_step, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "agent/yaw_bbox": (n_step, n_agent, 1),  # float32, yaw of the bbox heading
            "agent/yaw_rate": (n_step, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            # agent attributes
            "agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "agent/cmd": (n_agent, 8),  # bool one_hot
            "agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
            "agent/size": (n_agent, 3),  # float32: [length, width, height]
            "agent/goal": (n_agent, 4),  # float32: [x, y, theta, v]
            "agent/dest": (n_agent,),  # int64: index to map n_pl
            # map polylines
            "map/valid": (n_pl, n_pl_node),  # bool
            "map/type": (n_pl, 11),  # bool one_hot
            "map/pos": (n_pl, n_pl_node, 2),  # float32
            "map/dir": (n_pl, n_pl_node, 2),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # dummy traffic lights
            "tl_lane/valid": (n_step, n_tl),  # bool
            "tl_lane/state": (n_step, n_tl, 5),  # bool one_hot
            "tl_lane/idx": (n_step, n_tl),  # int, -1 means not valid
            "tl_stop/valid": (n_step, n_tl_stop),  # bool
            "tl_stop/state": (n_step, n_tl_stop, 5),  # bool one_hot
            "tl_stop/pos": (n_step, n_tl_stop, 2),  # x,y
            "tl_stop/dir": (n_step, n_tl_stop, 2),  # x,y
        }

        self.tensor_size_test = {
            # object_id for waymo metrics
            "history/agent/object_id": (n_agent,),
            "history/agent_no_sim/object_id": (n_agent_no_sim,),
            # agent_sim
            "history/agent/valid": (n_step_history, n_agent),  # bool,
            "history/agent/pos": (n_step_history, n_agent, 2),  # float32
            "history/agent/vel": (n_step_history, n_agent, 2),  # float32, v_x, v_y
            "history/agent/spd": (n_step_history, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "history/agent/acc": (n_step_history, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "history/agent/yaw_bbox": (n_step_history, n_agent, 1),  # float32, yaw of the bbox heading
            "history/agent/yaw_rate": (n_step_history, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            "history/agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "history/agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
            "history/agent/size": (n_agent, 3),  # float32: [length, width, height]
            # agent_no_sim not used by the models currently
            "history/agent_no_sim/valid": (n_step_history, n_agent_no_sim),
            "history/agent_no_sim/pos": (n_step_history, n_agent_no_sim, 2),
            "history/agent_no_sim/vel": (n_step_history, n_agent_no_sim, 2),
            "history/agent_no_sim/spd": (n_step_history, n_agent_no_sim, 1),
            "history/agent_no_sim/yaw_bbox": (n_step_history, n_agent_no_sim, 1),
            "history/agent_no_sim/type": (n_agent_no_sim, 3),
            "history/agent_no_sim/size": (n_agent_no_sim, 3),
            # map
            "map/valid": (n_pl, n_pl_node),  # bool
            "map/type": (n_pl, 11),  # bool one_hot
            "map/pos": (n_pl, n_pl_node, 2),  # float32
            "map/dir": (n_pl, n_pl_node, 2),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # dummy traffic_light
            "history/tl_lane/valid": (n_step_history, n_tl),  # bool
            "history/tl_lane/state": (n_step_history, n_tl, 5),  # bool one_hot
            "history/tl_lane/idx": (n_step_history, n_tl),  # int, -1 means not valid
            "history/tl_stop/valid": (n_step_history, n_tl_stop),  # bool
            "history/tl_stop/state": (n_step_history, n_tl_stop, 5),  # bool one_hot
            "history/tl_stop/pos": (n_step_history, n_tl_stop, 2),  # x,y
            "history/tl_stop/dir": (n_step_history, n_tl_stop, 2),  # dx,dy
        }

        self.tensor_size_val = {
            "agent/object_id": (n_agent,),
            "agent_no_sim/object_id": (n_agent_no_sim,),
            # agent_no_sim
            "agent_no_sim/valid": (n_step, n_agent_no_sim),  # bool,
            "agent_no_sim/pos": (n_step, n_agent_no_sim, 2),  # float32
            "agent_no_sim/vel": (n_step, n_agent_no_sim, 2),  # float32, v_x, v_y
            "agent_no_sim/spd": (n_step, n_agent_no_sim, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "agent_no_sim/yaw_bbox": (n_step, n_agent_no_sim, 1),  # float32, yaw of the bbox heading
            "agent_no_sim/type": (n_agent_no_sim, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "agent_no_sim/size": (n_agent_no_sim, 3),  # float32: [length, width, height]
        }

        self.tensor_size_val = {**self.tensor_size_val, **self.tensor_size_train, **self.tensor_size_test}

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if self.path_train_raster is not None:
                self.train_dataset = DatasetTrainRaster(
                    self.path_train_h5, 
                    self.tensor_size_train,
                    self.path_train_raster
                )
            else:
                self.train_dataset = DatasetTrain(self.path_train_h5, self.tensor_size_train)
                        
            if self.path_val_raster is not None:
                self.val_dataset = DatasetValRaster(
                    self.path_val_h5, 
                    self.tensor_size_val,
                    self.path_val_raster
                )
            else:
                self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
                
        elif stage == "validate":
            if self.path_val_raster is not None:
                self.val_dataset = DatasetValRaster(
                    self.path_val_h5, 
                    self.tensor_size_val,
                    self.path_val_raster
                )
            else:
                self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
                
        elif stage == "test":
            self.test_dataset = DatasetVal(self.path_test_h5, self.tensor_size_test)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.test_dataset, self.batch_size, self.num_workers)

    @staticmethod
    def custom_collate(batch):
        """Custom collate function to handle variable-sized tensors."""
        elem = batch[0]
        if isinstance(elem, dict):
            # Dictionary to store original dimensions for probability maps
            orig_dims = {}
            
            # First pass: collect original dimensions of important tensors
            for key in elem:
                if key == 'sdf_map' and isinstance(elem[key], (np.ndarray, torch.Tensor)):
                    # Store original dimensions of each SDF map in the batch
                    orig_dims['gt/sdf_map_orig_dims'] = torch.stack([
                        torch.tensor(d[key].shape[-2:], dtype=torch.long) for d in batch
                    ])

            # Normal collation
            result = {
                key: DataH5av2.custom_collate([d[key] for d in batch]) 
                if isinstance(elem[key], (np.ndarray, torch.Tensor)) 
                else default_collate([d[key] for d in batch])
                for key in elem
            }
            
            # Add original dimensions to the result
            result.update(orig_dims)
            return result
        elif isinstance(elem, (np.ndarray, torch.Tensor)):
            # Pad to max size in batch
            max_shape = tuple(max(s) for s in zip(*[b.shape for b in batch]))
            padded_batch = []
            
            for b in batch:
                if isinstance(b, np.ndarray):
                    # Preserve boolean dtype if input is boolean
                    dtype = torch.bool if b.dtype == np.bool_ else torch.float
                    b = torch.from_numpy(b).to(dtype)
                pad_sizes = [(0, m - s) for s, m in zip(b.shape, max_shape)]
                pad_sizes.reverse()  # torch.pad expects sizes in reverse order
                flat_pad = [item for sublist in pad_sizes for item in sublist]
                # Use same dtype for padding
                pad_value = False if b.dtype == torch.bool else 0
                padded = F.pad(b, flat_pad, value=pad_value)
                padded_batch.append(padded)
            
            return torch.stack(padded_batch)
        else:
            return default_collate(batch)

    def _get_dataloader(self, ds: Dataset, batch_size: int, num_workers: int) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=self.custom_collate,
        )
