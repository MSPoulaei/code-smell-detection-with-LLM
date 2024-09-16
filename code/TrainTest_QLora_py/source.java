
import java.util.ArrayList;
import java.util.List;

class Employee {
    private String name;
    private int age;
    private double salary;

    public Employee(String name, int age, double salary) {
        this.name = name;
        this.age = age;
        this.salary = salary;
    }

    public double calculateBonus() {
        double bonus;
        if (age > 50) {
            bonus = salary * 0.2;
        } else {
            bonus = salary * 0.1;
        }
        return bonus;
    }

    public void printEmployeeDetails() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
        System.out.println("Salary: " + salary);
        System.out.println("Bonus: " + calculateBonus());
    }

    public double getSalary() {
        return salary;
    }
}

public class Company {
    public static void main(String[] args) {
        Employee emp1 = new Employee("John", 45, 50000);
        Employee emp2 = new Employee("Jane", 55, 60000);
        Employee emp3 = new Employee("Doe", 40, 70000);

        emp1.printEmployeeDetails();
        emp2.printEmployeeDetails();
        emp3.printEmplong Method, Magic Number, Primitive Obsession, Data Clump, Feature Envy, Inappropriate Intimacy, TemporoyeeDetails();

        List<Employee> employees = new ArrayList<>();
        employees.add(emp1);
        employees.add(emp2);
        employees.add(emp3);

        System.out.println("Average Salary: " + calculateAverageSalary(employees));
    }

    public static double calculateAverageSalary(List<Employee> employees) {
        double totalSalary = 0;
        for (int i = 0; i < employees.size(); i++) {
            totalSalary += employees.get(i).getSalary();
        }
        return totalSalary / employees.size();
    }
}