import java.io.*;
import java.util.*;

public class back {
	
	//store the truth values so we can translate them into propositional atoms
	ArrayList<String[]> truths = new ArrayList<String[]>();
	ArrayList<String[]> atoms = new ArrayList<String[]>();

	public static void main(String[] args) throws Exception
	{
		back backend = new back();
		backend.read();
		backend.write();
	}
	
	//read from the input file
	void read() throws Exception
	{
		System.out.print("Enter path to input file:\n");
		Scanner kbd = new Scanner(System.in);
		String path = kbd.next();
		File file = new File(path);
		Scanner sc = new Scanner(file);
		String line = null;
		while(sc.hasNextLine())
		{
			while(!(line = sc.nextLine()).equals("0"))
			{
				String[] str = line.split("\\s");
				truths.add(str);
			}
			while(sc.hasNextLine())
			{
				line = sc.nextLine();
				String[] str = line.split("\\s");
				atoms.add(str);
			}
		}
	}
	
	//writes the correct path to the output file
	void write() throws Exception
	{
		System.out.print("Enter path to output file:\n");
		Scanner kbd = new Scanner(System.in);
		String path = kbd.next();
		File file = new File(path);
		if(!file.exists())
		{
			file.createNewFile();
		}
		
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		int index;
		for(int i = 0; i < truths.size(); i++)
		{
			if(truths.size() == 1)
			{
				bw.write(truths.get(0)[0] + " ");
				bw.write(truths.get(0)[1]);
				bw.newLine();
			}
		    else if(truths.get(i)[1].equals("T"))
			{
				index = Integer.parseInt(truths.get(i)[0]) - 1;
				if(atoms.get(index)[1].charAt(0)=='A'&&atoms.get(index)[1].charAt(1)=='t') {
					int character=3;
					while(Character.compare(atoms.get(index)[1].charAt(character),',')!=0) {	
						bw.write(atoms.get(index)[1].charAt(character));
						character++;
					}
					bw.newLine();
				}
			}
		}
		bw.close();
	}
}

