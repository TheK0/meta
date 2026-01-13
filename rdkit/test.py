from rdkit import Chem
from rdkit.Chem import AllChem
import os
import glob

def convert_mol_to_xyz(mol, name, output_dir):
    """将RDKit分子对象转换为XYZ文件"""
    if mol is None:
        print(f"  分子 {name} 读取失败")
        return False
    
    # 如果没有3D坐标,生成构象
    if not mol.GetNumConformers():
        mol = Chem.AddHs(mol)  # 加氢
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result == -1:
            print(f"  {name}: 3D构象生成失败")
            return False
        AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    
    # 写入XYZ文件
    lines = []
    lines.append(str(mol.GetNumAtoms()))  # 原子数
    lines.append(name)  # 注释行
    
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        lines.append(f"{atom.GetSymbol():<2} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
    
    xyz_path = os.path.join(output_dir, name + ".xyz")
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"  已保存: {xyz_path}")
    return True

def process_cdxml_file(cdxml_path, output_dir):
    """处理CDXML文件,提取所有分子"""
    print(f"处理CDXML文件: {cdxml_path}")
    base_name = os.path.splitext(os.path.basename(cdxml_path))[0]
    
    # 尝试读取CDXML文件
    try:
        # RDKit可以直接读取CDXML中的所有分子
        supplier = Chem.rdmolfiles.MolFromCDXMLFile(cdxml_path)
        if supplier:
            convert_mol_to_xyz(supplier, base_name, output_dir)
        else:
            print(f"  无法从 {cdxml_path} 读取分子")
    except Exception as e:
        print(f"  读取CDXML失败: {e}")

def process_mol_file(mol_path, output_dir):
    """处理单个MOL文件"""
    name = os.path.splitext(os.path.basename(mol_path))[0]
    print(f"处理MOL文件: {name}")
    
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    convert_mol_to_xyz(mol, name, output_dir)

def main():
    input_dir = r"C:\Users\yishengyuan\Downloads\ML\rdkit\mol_files"  # 输入文件夹
    output_dir = r"C:\Users\yishengyuan\Downloads\ML\rdkit\xyz_out"   # 输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有支持的文件格式
    file_patterns = {
        "*.mol": process_mol_file,
        "*.sdf": process_mol_file,
        "*.cdxml": process_cdxml_file,
    }
    
    total_processed = 0
    for pattern, processor in file_patterns.items():
        files = glob.glob(os.path.join(input_dir, pattern))
        for file_path in files:
            try:
                processor(file_path, output_dir)
                total_processed += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n总共处理了 {total_processed} 个文件")

if __name__ == "__main__":
    main()
