import json
def dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)
        f.close()


import numpy as np
import pandas as pd

class Material:
    def __init__(self, name, id, composition, 
                #  density, 
                 hydration=None):
        self.name = name
        self.id = id
        self.composition = composition
        self.hydration = hydration
        # self.density = density
    def __repr__(self):
        return f"Material({self.name}, {self.id}, {self.composition}, {self.hydration})"
    def __str__(self):
        mats = ' '.join([f"{key} {value}" for key, value in self.composition.items()])
        stack = f"m{self.id} {mats}\n"
        if self.hydration:
            stack += f"mt{self.id} lwtr{str(self.hydration)}\n"
        return stack
    def __getitem__(self, key):
        return self.composition[key]
    def __setitem__(self, key, value):
        self.composition[key] = value
    def __iter__(self):
        return iter(self.composition)
    def __len__(self):
        return len(self.composition)
    def __contains__(self, item):
        return item in self.composition
    def __eq__(self, other):
        return self.id == other.id
    def __ne__(self, other):
        return self.id != other.id
    def __lt__(self, other):
        return self.id < other.id
    def __le__(self, other):
        return self.id <= other.id
    def __gt__(self, other):
        return self.id > other.id
    def __ge__(self, other):
        return self.id >= other.id
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self):
        return {
            "name": self.name,
            "id": self.id,
            "composition": self.composition,
            "hydration": self.hydration
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["id"], data["composition"], data["hydration"])
    @classmethod
    def from_series(cls, series):
        return cls(series["name"], series["id"], series["composition"], series["hydration"])
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
    
    @classmethod
    def to_dataframe(cls, materials):
        return pd.DataFrame([m.to_dict() for m in materials]).set_index("id")
    
    @classmethod
    def to_json(cls, materials):
        return cls.to_dataframe(materials).to_json(orient="records")
    
    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))
    
    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, materials, filename):
        cls.to_dataframe(materials).to_csv(filename)

class surface():
    def __init__(self, surface_id, shape, params, group=None):
        self.surface_id = surface_id
        self.group = group
        self.shape = shape
        self.params = params
        self.position = params["pos"]
    def __repr__(self):
        return f"surface({self.surface_id}, {self.group}, {self.shape}, {self.params})"
    def __str__(self):
        return f"surface {self.surface_id}"
    def __getitem__(self, key):
        return self.params[key]
    def __setitem__(self, key, value):
        self.params[key] = value
    def __iter__(self):
        return iter(self.params)
    def __len__(self):
        return len(self.params)
    def __contains__(self, item):
        return item in self.params
    def __eq__(self, other):
        return self.surface_id == other.surface_id
    def __ne__(self, other):
        return self.surface_id != other.surface_id
    def __lt__(self, other):
        return self.surface_id < other.surface_id
    def __le__(self, other):
        return self.surface_id <= other.surface_id
    def __gt__(self, other):
        return self.surface_id > other.surface_id
    def __ge__(self, other):
        return self.surface_id >= other.surface_id
    def __hash__(self):
        return hash(self.surface_id)
    
    def to_dict(self):
        return {
            "surface_id": self.surface_id,
            "group": self.group,
            "shape": self.shape,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["surface_id"], data["group"], data["shape"], data["params"])
    
    @classmethod
    def from_series(cls, series):
        return cls(series["surface_id"], series["group"], series["shape"], series["params"])
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
    
    @classmethod
    def to_dataframe(cls, surfaces):
        return pd.DataFrame([s.to_dict() for s in surfaces]).set_index("surface_id")
    
    @classmethod
    def to_json(cls, surfaces):
        return cls.to_dataframe(surfaces).to_json(orient="records")
    
    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))
    
    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, surfaces, filename):
        cls.to_dataframe(surfaces).to_csv(filename)

class sphere(surface):
    def __init__(self, surface_id, pos, r, group=None):
        self.pos = pos
        self.r = r
        params = {
            "pos": pos,
            "r": r
        }
        super().__init__(surface_id, "sphere", params, group)
    def __repr__(self):
        return f"sphere({self.surface_id}, {self.group}, {self.params})"
    def __str__(self):
        return f"{self.surface_id} sph {' '.join([str(p) for p in self.pos])} {self.r}\n"
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return (v / norm).tolist()

class cylinder(surface):
    def __init__(self, surface_id, pos, r, height, dir, group=None):
        self.pos = pos
        self.r = r
        self.height = height
        self.dir = normalize(dir)
        self.group = group
        self.params = {
            "pos": pos,
            "r": r,
            "height": height,
            "dir": dir
        }
        super().__init__(surface_id, "cylinder", self.params, group)
    def __repr__(self):
        return f"cylinder({self.surface_id}, {' '.join([str(p)+'='+str(self.params[p]) for p in self.params.keys()])}"
    def __str__(self):
        
        return f"{self.surface_id} rcc {' '.join([str(p) for p in self.pos])} {' '.join([str(d*self.height) for d in self.dir])} {self.r}\n"
    
class box(surface):
    def __init__(self, surface_id, pos, l, w, h, group=None):
        self.pos = pos
        self.l = l #x
        self.w = w #y
        self.h = h #z
        self.params = {
            "pos": pos,
            "l": l,
            "w": w,
            "h": h
        }
        super().__init__(surface_id, "box", self.params, group)
    def __repr__(self):
        return f"box({self.surface_id}, {self.group}, {self.params})"
    def __str__(self):
        corner = [p - l/2 for p, l in zip(self.pos, [self.l, self.w, self.h])]
        return f"{self.surface_id} box {' '.join([str(p) for p in corner])} {self.l} 0 0 0 {self.w} 0 0 0 {self.h}\n"
    
class rpp(surface):
    def __init__(self, surface_id, min, max, group=None):
        self.min = min
        self.max = max
        self.params = {
            "pos": min,
            "size": max
        }
        super().__init__(surface_id, "rcc", self.params, group)
    def __repr__(self):
        return f"rpp({self.surface_id}, {self.group}, {self.params})"
    def __str__(self):
        # zip and flatten
        l = [p for pair in zip(self.min, self.max) for p in pair]
        return f"{self.surface_id} rpp {' '.join([str(p) for p in l])}\n"

class cell():
    def __init__(self, cell_id, name, material, density, surfaces, importance=None):
        self.cell_id = cell_id
        self.name = name
        self.material = material
        self.density = density
        self.surfaces = surfaces
        self.importance = importance
    def __repr__(self):
        return f"cell({self.cell_id}, {self.name}, {self.material}, {self.density}, {self.surfaces}, {self.importance})"
    def __str__(self):
        stack = f"{self.cell_id} {self.material} "
        if self.material != 0:
            stack += f"{self.density} "
            
        stack += f"{self.surfaces} "
        if self.importance:
            stack += f"{self.importance}"
        stack += "\n"
        return stack
    def __getitem__(self, key):
        return self.surfaces[key]
    def __setitem__(self, key, value):
        self.surfaces[key] = value
    def __iter__(self):
        return iter(self.surfaces)
    def __len__(self):
        return len(self.surfaces)
    def __contains__(self, item):
        return item in self.surfaces
    def __eq__(self, other):
        return self.cell_id == other.cell_id
    def __ne__(self, other):
        return self.cell_id != other.cell_id
    def __lt__(self, other):
        return self.cell_id < other.cell_id
    def __le__(self, other):
        return self.cell_id <= other.cell_id
    def __gt__(self, other):
        return self.cell_id > other.cell_id
    def __ge__(self, other):
        return self.cell_id >= other.cell_id
    def __hash__(self):
        return hash(self.cell_id)
    
    def to_dict(self):
        return {
            "cell_id": self.cell_id,
            "material": self.material,
            "density": self.density,
            "surfaces": self.surfaces,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["cell_id"], data["material"], data["density"], data["surfaces"], data["importance"])
    
    @classmethod
    def from_series(cls, series):
        return cls(series["cell_id"], series["material"], series["density"], series["surfaces"], series["importance"])
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
        
    @classmethod
    def to_dataframe(cls, cells):
        return pd.DataFrame([c.to_dict() for c in cells]).set_index("cell_id")
    
    @classmethod
    def to_json(cls, cells):
        return cls.to_dataframe(cells).to_json(orient="records")
    
    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))
    
    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, cells, filename):
        cls.to_dataframe(cells).to_csv(filename)

class source():
    def __init__(self, source_id, pos, energy, si=None, sp=None):
        self.source_id = source_id
        self.pos = pos
        self.energy = energy
        self.si = si
        self.sp = sp
    def __repr__(self):
        return f"source({self.source_id}, {self.pos}, {self.energy})"
    def __str__(self):
        out = f"sdef pos={' '.join([str(i) for i in self.pos])} erg={self.energy}\n"
        if self.si:
            out += f"si{self.source_id} {self.si}\n"
        if self.sp:
            out += f"sp{self.source_id} {self.sp}\n"
        return out
    def __hash__(self):
        return hash(self.source_id)
    
    def to_dict(self):
        return {
            "source_id": self.source_id,
            "pos": self.pos,
            "energy": self.energy
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["source_id"], data["pos"], data["energy"])
    
    @classmethod
    def from_series(cls, series):
        return cls(series["source_id"], series["pos"], series["energy"])
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
    
    @classmethod
    def to_dataframe(cls, sources):
        return pd.DataFrame([s.to_dict() for s in sources]).set_index("source_id")
    
    @classmethod
    def to_json(cls, sources):
        return cls.to_dataframe(sources).to_json(orient="records")
    

    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))
    
    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, sources, filename):
        cls.to_dataframe(sources).to_csv(filename)

class tally():
    def __init__(self, tally_id, tally_type, particle, cells, energy_bins=None, time_bins=None, geb = None, tmc = None, phl = None, tag = None):
        self.tally_id = tally_id
        self.tally_type = tally_type
        self.particle = particle
        self.cells = cells
        self.energy_bins = energy_bins # string or list or np.array
        self.time_bins = time_bins # string or list or np.array

        self.geb = geb
        self.tmc = tmc
        self.phl = phl
        self.tag = tag

    def __repr__(self):
        return f"tally({self.tally_id}, {self.tally_type}, {self.particle}, {self.cells}, {self.energy_bins}, {self.time_bins})"
    def __str__(self):
        stack = f"F{self.tally_id}:{self.particle} {self.cells}\n"
        if self.energy_bins:
            if isinstance(self.energy_bins, str):
                parsed_bins = self.energy_bins
            elif isinstance(self.energy_bins, list) or isinstance(self.energy_bins, np.ndarray):
                parsed_bins = ' '.join([str(bin) for bin in self.energy_bins])
            stack += f"e{self.tally_id} {parsed_bins}\n"
        if self.time_bins:
            if isinstance(self.time_bins, str):
                parsed_bins = self.time_bins
            elif isinstance(self.time_bins, list) or isinstance(self.time_bins, np.ndarray):
                parsed_bins = ' '.join([str(bin) for bin in self.time_bins])
            stack += f"T{self.tally_id} {parsed_bins}\n"
        if self.geb or self.tmc or self.phl or self.tag:
            stack += f"FT{self.tally_id} "
            if self.phl:
                stack += f"phl 1 {self.phl} 1 0 "
            if self.geb:
                stack += f"geb {self.geb[0]} {self.geb[1]} {self.geb[2]} "
            if self.tmc:
                stack += f"tmc {self.tmc[0]} {self.tmc[1]} "
            if self.tag:
                stack += f"tag {self.tag} "
            stack += "\n"
        return stack
    def __getitem__(self, key):
        return self.cells[key]
    def __setitem__(self, key, value):
        self.cells[key] = value
    def __iter__(self):
        return iter(self.cells)
    def __len__(self):
        return len(self.cells)
    def __contains__(self, item):
        return item in self.cells
    def __eq__(self, other):
        return self.tally_id == other.tally_id
    def __ne__(self, other):
        return self.tally_id != other.tally_id
    def __lt__(self, other):
        return self.tally_id < other.tally_id
    def __le__(self, other):
        return self.tally_id <= other.tally_id
    def __gt__(self, other):
        return self.tally_id > other.tally_id
    def __ge__(self, other):
        return self.tally_id >= other.tally_id
    def __hash__(self):
        return hash(self.tally_id)
    
    def to_dict(self):
        return {
            "tally_id": self.tally_id,
            "tally_type": self.tally_type,
            "particle": self.particle,
            "cells": self.cells,
            "energy_bins": self.energy_bins,
            "time_bins": self.time_bins,
            "energy_range": self.energy_range,
            "time_range": self.time_range
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["tally_id"], data["tally_type"], data["particle"], data["cells"], data["energy_bins"], data["time_bins"], data["energy_range"], data["time_range"])
    
    @classmethod
    def from_series(cls, series):
        return cls(series["tally_id"], series["tally_type"], series["particle"], series["cells"], series["energy_bins"], series["time_bins"], series["energy_range"], series["time_range"])
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
    
    @classmethod
    def to_dataframe(cls, tallies):
        return pd.DataFrame([t.to_dict() for t in tallies]).set_index("tally_id")
    
    @classmethod
    def to_json(cls, tallies):
        return cls.to_dataframe(tallies).to_json(orient="records")
    
    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))
    
    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, tallies, filename):
        cls.to_dataframe(tallies).to_csv(filename)

class sim():
    def __init__(self, title, nps=1e5, prdmp=None, seed=1337, source=None, tallies=None, surfaces=None, cells=None, materials=None, cont=False):
        self.title = title
        self.nps = int(nps)
        self.prdmp = prdmp
        self.seed = seed


        # stuff thats already an object
        self.source = source
        self.tallies = tallies
        self.surfaces = surfaces
        self.cells = cells
        self.materials = materials
        self.cont = cont
    def __str__(self):
        stack = f""
        if self.cont:
            stack += f"continue ${self.title}\n"
        else:
            stack += f"{self.title}\n"
        if self.cells:
            stack += ''.join([str(cell) for cell in self.cells])
            stack += "\n"
        if self.surfaces:
            stack += ''.join([str(surface) for surface in self.surfaces])
            stack += "\n"
            
        if self.materials:
            stack += ''.join([str(material) for material in self.materials])
        if self.source:
            stack += str(self.source)
        if self.tallies:
            stack += ''.join([str(tally) for tally in self.tallies])
        
        if self.prdmp:
            stack += f"prdmp {' '.join([str(p) for p in self.prdmp])}\n"
        if self.nps:
            stack += f"nps {self.nps}\n"
        if self.seed:
            stack += f"rand seed = {self.seed}\n"
        stack+="mode n p e \nphys:n 1j 14\nphys:p\nphys:e"
        return stack
    def __getitem__(self, key):
        return self.tallies[key]
    def __setitem__(self, key, value):
        self.tallies[key] = value
    def __hash__(self):
        return hash(self.title)
    
    @classmethod
    def from_dataframe(cls, df):
        return [cls.from_series(df.loc[i]) for i in df.index]
    
    @classmethod
    def to_dataframe(cls, miscs):
        return pd.DataFrame([m.to_dict() for m in miscs]).set_index("title")
    
    @classmethod
    def to_json(cls, miscs):
        return cls.to_dataframe(miscs).to_json(orient="records")

    @classmethod
    def from_json(cls, json_data):
        return cls.from_dataframe(pd.read_json(json_data))

    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))
    
    @classmethod
    def to_csv(cls, miscs, filename):
        cls.to_dataframe(miscs).to_csv(filename)

